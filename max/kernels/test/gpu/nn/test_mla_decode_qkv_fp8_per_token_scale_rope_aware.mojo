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

"""Test for SnapMLA FP8 per-token-scale rope-aware MLA decode through the full
production dispatch path (split-K + combine kernel).

This test exercises the complete production dispatch path:
  flare_mla_decoding[per_token_scale_rope_aware=True]
    -> flare_mla_decoding_dispatch[per_token_scale_rope_aware=True]
      -> mla_decode_sm100_dispatch[per_token_scale_rope_aware=True]
        -> _mla_decode_sm100_dispatch_impl
          -> mla_decode_sm100_sink_split_k[per_token_scale_rope_aware=True]
            -> creates 4 TMAs (Q_nope, Q_rope, K_content, K_rope)
            -> launch_mla_sm100_decode_fp8_per_token_scale_rope_aware
          + combine kernel for split-K reduction

Using a PagedKVCache (not LayoutTensorMHAOperand) because the production
path's create_rope_tma_tile is only implemented for PagedKVCache.

Layout:
  - Q: [total_q_tokens, num_heads, 640] float8_e4m3fn
    - bytes 0..511: FP8 content (512 nope dims)
    - bytes 512..639: BF16 rope (64 rope dims = 128 bytes)
  - KV cache: [pages, 1, layers, page_size, kv_heads, 640] float8_e4m3fn
    - same interleaved layout per token row
  - Output: [total_q_tokens, num_heads, 512] bfloat16
"""

from std.math import ceildiv, nan
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator

from max.gpu.host import DeviceContext
from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
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
# Test parameters matching DeepSeek-V2/V3 rope-aware FP8 path
# ===-----------------------------------------------------------------------===#

comptime LOGICAL_DEPTH = 576  # Logical Q/K head dimension (512 nope + 64 rope)
comptime V_DEPTH = 512  # Output head dimension (nope only)
comptime ROPE_DIM = 64  # Rope dimension
comptime PHYSICAL_DIM = 640  # Bytes per row: 512 FP8 + 128 BF16 = 640
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1  # Single layer for testing
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


# ===-----------------------------------------------------------------------===#
# Helper: create interleaved FP8+BF16 data on host
# ===-----------------------------------------------------------------------===#


def create_interleaved_q_data(
    q_host_fp8: MutPointer[Float8_e4m3fn, _],
    q_ref_bf16: MutPointer[BFloat16, _],
    total_q_tokens: Int,
    num_heads: Int,
) raises:
    """Create interleaved Q data and the dequantized BF16 reference."""
    # Generate random BF16 content and rope data
    var content_size = total_q_tokens * num_heads * V_DEPTH
    var rope_size = total_q_tokens * num_heads * ROPE_DIM
    var content_bf16 = List(length=content_size, fill=BFloat16(0))
    var rope_bf16 = List(length=rope_size, fill=BFloat16(0))
    randn(
        content_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    randn(
        rope_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )

    for t in range(total_q_tokens):
        for h in range(num_heads):
            var row_idx = t * num_heads + h
            var fp8_row_start = row_idx * PHYSICAL_DIM
            var ref_row_start = row_idx * LOGICAL_DEPTH
            var content_start = row_idx * V_DEPTH
            var rope_start = row_idx * ROPE_DIM

            # Content: quantize BF16 -> FP8, store in interleaved, dequant for ref
            for d in range(V_DEPTH):
                var val = content_bf16[content_start + d]
                var fp8_val = val.cast[.float8_e4m3fn]()
                q_host_fp8[fp8_row_start + d] = fp8_val
                # Dequantized for reference
                q_ref_bf16[ref_row_start + d] = fp8_val.cast[.bfloat16]()

            # Rope: BF16 stored as raw bytes in the FP8 buffer, and in ref
            var rope_byte_ptr = (q_host_fp8 + fp8_row_start + V_DEPTH).bitcast[
                BFloat16
            ]()
            for d in range(ROPE_DIM):
                rope_byte_ptr[d] = rope_bf16[rope_start + d]
                q_ref_bf16[ref_row_start + V_DEPTH + d] = rope_bf16[
                    rope_start + d
                ]


def create_interleaved_kv_block_data(
    blocks_host: MutPointer[Float8_e4m3fn, _],
    block_elems: Int,
    total_pages: Int,
    page_size: Int,
) raises:
    """Fill KV cache blocks with interleaved FP8+BF16 data.

    Each token row is 640 FP8 elements (bytes):
      - bytes 0..511: random FP8 content
      - bytes 512..639: random BF16 rope (64 bf16 = 128 bytes)

    Uses small std dev to keep dot products moderate.
    """
    # Generate random BF16 for all content and rope, then pack
    var num_tokens = total_pages * page_size * KV_NUM_HEADS
    var content_bf16 = List(length=num_tokens * V_DEPTH, fill=BFloat16(0))
    var rope_bf16 = List(length=num_tokens * ROPE_DIM, fill=BFloat16(0))
    randn(
        content_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    randn(
        rope_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )

    for tok in range(num_tokens):
        var fp8_row_start = tok * PHYSICAL_DIM
        var content_start = tok * V_DEPTH
        var rope_start = tok * ROPE_DIM

        # Content: BF16 -> FP8
        for d in range(V_DEPTH):
            blocks_host[fp8_row_start + d] = content_bf16[
                content_start + d
            ].cast[.float8_e4m3fn]()

        # Rope: store BF16 bytes in the FP8 buffer
        var rope_byte_ptr = (blocks_host + fp8_row_start + V_DEPTH).bitcast[
            BFloat16
        ]()
        for d in range(ROPE_DIM):
            rope_byte_ptr[d] = rope_bf16[rope_start + d]


def extract_bf16_kv_from_block(
    blocks_host: ImmPointer[Float8_e4m3fn, _],
    k_bf16_out: MutPointer[BFloat16, _],
    physical_page: Int,
    tok_in_page: Int,
    kv_dim2: Int,
    page_stride: Int,
) raises:
    """Extract one token's full BF16 data (576 dims) from a paged KV cache block.

    Dequantizes FP8 content (512 dims) and copies BF16 rope (64 dims).
    """
    var src_base = (
        physical_page * page_stride + tok_in_page * KV_NUM_HEADS * PHYSICAL_DIM
    )

    for kh in range(KV_NUM_HEADS):
        var src_row = src_base + kh * PHYSICAL_DIM
        var dst_row = kh * LOGICAL_DEPTH

        # Dequant FP8 content -> BF16
        for d in range(V_DEPTH):
            k_bf16_out[dst_row + d] = blocks_host[src_row + d].cast[
                DType.bfloat16
            ]()

        # Copy BF16 rope
        var rope_src = (blocks_host + src_row + V_DEPTH).bitcast[BFloat16]()
        for d in range(ROPE_DIM):
            k_bf16_out[dst_row + V_DEPTH + d] = rope_src[d]


# ===-----------------------------------------------------------------------===#
# Core test function
# ===-----------------------------------------------------------------------===#


def run_test[
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    comptime fp8_type = DType.float8_e4m3fn
    comptime bf16_type = DType.bfloat16
    comptime group = num_heads
    comptime scale = Float32(0.125)
    comptime q_max_seq_len = 1

    var batch_size = len(cache_lengths)
    var total_q_tokens = batch_size  # seq_len=1 per batch

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    # -------------------------------------------------------------------
    # Step 1: Create paged KV cache with interleaved FP8+BF16 layout
    # -------------------------------------------------------------------
    # head_size=640 because each token row is 640 bytes in FP8 terms
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=PHYSICAL_DIM, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True

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

    var blocks_host = ctx.enqueue_create_host_buffer[fp8_type](block_elems)

    # Fill with interleaved FP8 content + BF16 rope data
    create_interleaved_kv_block_data(
        blocks_host.unsafe_ptr(), block_elems, total_pages, PAGE_SIZE
    )

    # Zero out tail slots in each page (tokens beyond num_keys)
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = Scalar[fp8_type](0)
            cur_page += 1

    # -------------------------------------------------------------------
    # Step 1b: Create per-token KV scales tensor
    # -------------------------------------------------------------------
    # With per_token_scale_rope_aware, the kernel always has
    # has_per_token_scales=True and loads KV scales from the cache.
    # Valid tokens get 1.0; unused/padding slots get NaN to
    # deterministically catch OOB scale reads in the kernel.
    # (The kernel must sanitize with max(scale, 0) to handle this.)
    comptime head_dim_gran = 1  # ceildiv(PHYSICAL_DIM, PHYSICAL_DIM)
    var scales_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        head_dim_gran,
    )
    var scales_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * head_dim_gran
    )
    # Initialize ALL scale slots to NaN (poison unused slots).
    # Valid token scales get overwritten below.
    var scales_host = ctx.enqueue_create_host_buffer[.float32](scales_elems)
    for i in range(scales_elems):
        scales_host[i] = nan[.float32]()
    # Then set valid token scales to 1.0
    var _scale_page_stride = (
        kv_dim2 * NUM_LAYERS * PAGE_SIZE * kv_params.num_heads * head_dim_gran
    )
    var _cur_page_s = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            for tok_in_page in range(valid_toks):
                var offset = (
                    _cur_page_s * _scale_page_stride
                    + tok_in_page * kv_params.num_heads * head_dim_gran
                )
                scales_host[offset] = Float32(1.0)
            _cur_page_s += 1

    # -------------------------------------------------------------------
    # Step 1c: Create per-Q-token scales (all 1.0)
    # -------------------------------------------------------------------
    var q_scales_host = ctx.enqueue_create_host_buffer[.float32](total_q_tokens)
    for i in range(total_q_tokens):
        q_scales_host[i] = Float32(1.0)

    # Cache lengths and lookup table
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # -------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, 640] FP8)
    # -------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * PHYSICAL_DIM
    var q_host_fp8 = ctx.enqueue_create_host_buffer[fp8_type](q_size)
    # Full BF16 reference Q: [total_q_tokens, num_heads, 576]
    var q_ref_size = total_q_tokens * num_heads * LOGICAL_DEPTH
    var q_ref_bf16 = ctx.enqueue_create_host_buffer[bf16_type](q_ref_size)

    create_interleaved_q_data(
        q_host_fp8.unsafe_ptr(),
        q_ref_bf16.unsafe_ptr(),
        total_q_tokens,
        num_heads,
    )

    # -------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -------------------------------------------------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -------------------------------------------------------------------
    # Step 4: Output tensor [total_q_tokens, num_heads, V_DEPTH=512] BF16
    # -------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = ctx.enqueue_create_host_buffer[bf16_type](out_size)

    # -------------------------------------------------------------------
    # Step 5: Copy to device
    # -------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[fp8_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    ctx.enqueue_copy(scales_device, scales_host)

    var q_scales_device = ctx.enqueue_create_buffer[.float32](total_q_tokens)
    ctx.enqueue_copy(q_scales_device, q_scales_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[fp8_type](q_size)
    ctx.enqueue_copy(q_device, q_host_fp8)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[bf16_type](out_size)

    ctx.synchronize()

    # -------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection
    # -------------------------------------------------------------------
    var blocks_lt = LayoutTensor[fp8_type, Layout.row_major[6]()](
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
        fp8_type,
        kv_params,
        PAGE_SIZE,
        scale_dtype_=DType.float32,
        quantization_granularity_=PHYSICAL_DIM,
    ](
        LayoutTensor[fp8_type, Layout.row_major[6]()](
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
        LayoutTensor[.float32, Layout.row_major[6]()](
            scales_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                scales_lt.runtime_layout.shape.value,
                scales_lt.runtime_layout.stride.value,
            ),
        ),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # Q: [total_q_tokens, num_heads, 640] float8_e4m3fn
    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[PHYSICAL_DIM])),
    )

    # Output: [total_q_tokens, num_heads, V_DEPTH=512] bfloat16
    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged layout
    var row_offsets_tt = TileTensor(
        row_offsets_device,
        row_major(batch_size + 1),
    )

    # q_scale_ptr: reinterpret as Pointer with MutAnyOrigin
    var q_scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        q_scales_device.unsafe_ptr()
    )

    # -------------------------------------------------------------------
    # Step 7: Call the full production dispatch path
    # -------------------------------------------------------------------
    print("  Launching MLA decode kernel (full dispatch path)...")

    # Pre-compute scalar dispatch args (batch_size, max_cache_len, q_max_seq_len).
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[fp8_type](num_heads, LOGICAL_DEPTH),
        ragged=True,
        per_token_scale_rope_aware=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf=scalar_args_buf_tt,
        q_max_seq_len=q_max_seq_len,
        q_scale_ptr=q_scale_ptr,
    )

    ctx.synchronize()
    print("  Kernel completed.")
    _ = mla_args^

    # -------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive per batch
    # -------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    var rtol = 5e-2
    var atol = 3e-1
    var total_checked = 0
    var max_abs_err = Float64(0)

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var ref_num_keys = cache_len + q_max_seq_len

        # Extract contiguous BF16 K for this batch from paged blocks
        # K shape: [ref_num_keys, KV_NUM_HEADS, LOGICAL_DEPTH=576]
        var k_b_size = ref_num_keys * KV_NUM_HEADS * LOGICAL_DEPTH
        var k_b_host = ctx.enqueue_create_host_buffer[bf16_type](k_b_size)

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var dst_offset = tok * KV_NUM_HEADS * LOGICAL_DEPTH
            extract_bf16_kv_from_block(
                blocks_host.unsafe_ptr(),
                k_b_host.unsafe_ptr() + dst_offset,
                physical_page,
                tok_in_page,
                kv_dim2,
                _page_stride,
            )

        # Q for this batch: BF16 [1, 1, num_heads, LOGICAL_DEPTH=576]
        var q_b_size = 1 * num_heads * LOGICAL_DEPTH
        var q_b_host = ctx.enqueue_create_host_buffer[bf16_type](q_b_size)
        for i in range(q_b_size):
            q_b_host[i] = q_ref_bf16[b * num_heads * LOGICAL_DEPTH + i]

        # Reference output: [1, 1, num_heads, LOGICAL_DEPTH=576] (full depth)
        var ref_b_size = 1 * num_heads * LOGICAL_DEPTH
        var ref_b_host = ctx.enqueue_create_host_buffer[bf16_type](ref_b_size)

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[bf16_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[bf16_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[bf16_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors for mha_gpu_naive reference
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[LOGICAL_DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[LOGICAL_DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[LOGICAL_DEPTH])),
        )

        # mha_gpu_naive: K used as both K and V (MLA: V = K[:,:,:512])
        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            NullMask(),
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            1,  # seq_len
            ref_num_keys,
            num_heads,
            LOGICAL_DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims per head
        var out_offset = b * num_heads * V_DEPTH
        var nan_count = 0
        var first_nan_h = -1
        var first_nan_d = -1
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host[d + LOGICAL_DEPTH * h].cast[
                    DType.float64
                ]()
                var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                    DType.float64
                ]()
                if actual != actual:  # NaN check
                    nan_count += 1
                    if first_nan_h == -1:
                        first_nan_h = h
                        first_nan_d = d
                        print(
                            "  NaN at batch",
                            b,
                            "head",
                            h,
                            "depth",
                            d,
                            "expect",
                            expect,
                        )
                    continue
                var abs_err = abs(actual - expect)
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if abs_err > 1e-1:
                    print(b, h, d, actual, expect)
        if nan_count > 0:
            print(
                "  TOTAL NaNs for batch",
                b,
                ":",
                nan_count,
                "out of",
                num_heads * V_DEPTH,
            )
            print(
                "  First NaN at h=",
                first_nan_h,
                "d=",
                first_nan_d,
            )
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host[d + LOGICAL_DEPTH * h].cast[
                    DType.float64
                ]()
                var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                    DType.float64
                ]()
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

        total_checked += num_heads * V_DEPTH

        # Cleanup per-batch buffers
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

    # Cleanup

    _ = blocks_device
    _ = scales_device
    _ = q_scales_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


# ===-----------------------------------------------------------------------===#
# Test with NON-TRIVIAL per-token scales (sigma_KV != 1.0, sigma_Q != 1.0)
# ===-----------------------------------------------------------------------===#


def _scale_palette(index: Int) -> Float32:
    """Pick a non-trivial scale from a palette by index (wrapping).

    Uses power-of-2 values to avoid additional quantization noise in the
    reference computation.  These are exact in float32 and FP8 e8m0.
    """
    if index % 7 == 0:
        return 0.25
    if index % 7 == 1:
        return 0.5
    if index % 7 == 2:
        return 1.5
    if index % 7 == 3:
        return 2.0
    if index % 7 == 4:
        return 0.75
    if index % 7 == 5:
        return 1.25
    return 4.0


def run_test_with_scales[
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    """Test MLA decode with non-trivial per-token scales.

    Creates:
      - sigma_KV[t]: per-KV-token scale from palette, stored in KV cache scales
      - sigma_Q[b]:  per-Q-token scale from palette, passed via q_scale_ptr

    Reference computation applies these scales to dequantized BF16 data
    before calling mha_gpu_naive, so the reference matches the kernel's
    scale-aware computation.
    """
    comptime fp8_type = DType.float8_e4m3fn
    comptime bf16_type = DType.bfloat16
    comptime group = num_heads
    comptime scale = Float32(0.125)
    comptime q_max_seq_len = 1

    var batch_size = len(cache_lengths)
    var total_q_tokens = batch_size  # seq_len=1 per batch

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " (with non-trivial scales)",
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    # -------------------------------------------------------------------
    # Step 1: Create paged KV cache with interleaved FP8+BF16 layout
    #         AND per-token scales (quantization_granularity = PHYSICAL_DIM)
    # -------------------------------------------------------------------
    # With quantization_granularity=PHYSICAL_DIM, head_dim_gran = ceildiv(640, 640) = 1,
    # giving exactly 1 float32 scale per token.
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=PHYSICAL_DIM, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True

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

    var blocks_host = ctx.enqueue_create_host_buffer[fp8_type](block_elems)

    # Fill with interleaved FP8 content + BF16 rope data
    create_interleaved_kv_block_data(
        blocks_host.unsafe_ptr(), block_elems, total_pages, PAGE_SIZE
    )

    # Zero out tail slots in each page (tokens beyond num_keys)
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = Scalar[fp8_type](0)
            cur_page += 1

    # -------------------------------------------------------------------
    # Step 1b: Create per-token KV scales tensor
    # -------------------------------------------------------------------
    # Scales shape: [total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, num_heads, head_dim_gran]
    # With head_dim_gran=1: one float32 per token.
    comptime head_dim_gran = 1  # ceildiv(PHYSICAL_DIM, PHYSICAL_DIM)
    var scales_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        head_dim_gran,
    )
    var scales_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * head_dim_gran
    )
    # Initialize ALL scale slots to NaN (poison unused slots to
    # deterministically catch OOB scale reads in the kernel).
    var scales_host = ctx.enqueue_create_host_buffer[.float32](scales_elems)
    for i in range(scales_elems):
        scales_host[i] = nan[.float32]()

    # Fill valid token slots with non-trivial per-token scales.
    # scale(token) = palette[(page * PAGE_SIZE + tok_in_page) * 3]
    var scale_page_stride = (
        kv_dim2 * NUM_LAYERS * PAGE_SIZE * kv_params.num_heads * head_dim_gran
    )
    cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            for tok_in_page in range(valid_toks):
                var offset = (
                    cur_page * scale_page_stride
                    + tok_in_page * kv_params.num_heads * head_dim_gran
                )
                var global_tok = pg * PAGE_SIZE + tok_in_page
                scales_host[offset] = _scale_palette(
                    (bi * 1000 + global_tok) * 3
                )
            cur_page += 1

    # -------------------------------------------------------------------
    # Step 1c: Create per-Q-token scales (sigma_Q)
    # -------------------------------------------------------------------
    var q_scales_host = ctx.enqueue_create_host_buffer[.float32](total_q_tokens)
    for i in range(total_q_tokens):
        q_scales_host[i] = _scale_palette((i + 42) * 5)
    print("  sigma_Q values:")
    for i in range(total_q_tokens):
        print("    batch", i, ": sigma_Q=", q_scales_host[i])

    # -------------------------------------------------------------------
    # Step 1d: Scale Domain Alignment (SnapMLA Eq. 6)
    #
    # Pre-divide K_rope[t] by sigma_KV[t] for each KV token t.
    # This makes the kernel's uniform application of sigma_KV[t] to the
    # combined (content + rope) score mathematically exact:
    #   sigma_KV[t] * (Q_nope·K_nope + Q_rope_aligned·K_rope_aligned)
    #   = sigma_KV[t]*Q_nope·K_nope + Q_rope·K_rope
    # because K_rope_aligned = K_rope / sigma_KV[t], so sigma_KV cancels.
    # -------------------------------------------------------------------
    cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            for tok_in_page in range(valid_toks):
                # Get sigma_KV[t] for this token
                var s_offset = (
                    cur_page * scale_page_stride
                    + tok_in_page * kv_params.num_heads * head_dim_gran
                )
                var sigma_kv_t = scales_host[s_offset]

                # Divide K_rope BF16 values by sigma_KV[t]
                var tok_fp8_base = (
                    cur_page * _page_stride
                    + tok_in_page * kv_params.num_heads * PHYSICAL_DIM
                )
                for kh in range(KV_NUM_HEADS):
                    var rope_ptr = (
                        blocks_host.unsafe_ptr()
                        + tok_fp8_base
                        + kh * PHYSICAL_DIM
                        + V_DEPTH
                    ).bitcast[BFloat16]()
                    for d in range(ROPE_DIM):
                        rope_ptr[d] = (
                            rope_ptr[d].cast[.float32]() / sigma_kv_t
                        ).cast[bf16_type]()
            cur_page += 1

    # Cache lengths and lookup table
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_pages_per_batch = ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # -------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, 640] FP8)
    # -------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * PHYSICAL_DIM
    var q_host_fp8 = ctx.enqueue_create_host_buffer[fp8_type](q_size)
    # Full BF16 reference Q: [total_q_tokens, num_heads, 576]
    var q_ref_size = total_q_tokens * num_heads * LOGICAL_DEPTH
    var q_ref_bf16 = ctx.enqueue_create_host_buffer[bf16_type](q_ref_size)

    create_interleaved_q_data(
        q_host_fp8.unsafe_ptr(),
        q_ref_bf16.unsafe_ptr(),
        total_q_tokens,
        num_heads,
    )

    # -------------------------------------------------------------------
    # Step 2b: Scale Domain Alignment for Q_rope (SnapMLA Eq. 6)
    #
    # Pre-divide Q_rope[q] by sigma_Q[q] for each Q token q.
    # Together with K_rope pre-division (Step 1d), this makes the
    # kernel's uniform scaling mathematically exact:
    #   sigma_Q * sigma_KV[t] * (Q_nope·K_nope + Q_rope_aligned·K_rope_aligned)
    #   = sigma_Q * sigma_KV[t] * Q_nope·K_nope + Q_rope·K_rope
    # -------------------------------------------------------------------
    for t in range(total_q_tokens):
        var sigma_q_t = q_scales_host[t]
        for h in range(num_heads):
            var row_idx = t * num_heads + h
            # Divide Q_rope BF16 in the interleaved FP8 buffer
            var fp8_rope_ptr = (
                q_host_fp8.unsafe_ptr() + row_idx * PHYSICAL_DIM + V_DEPTH
            ).bitcast[BFloat16]()
            for d in range(ROPE_DIM):
                fp8_rope_ptr[d] = (
                    fp8_rope_ptr[d].cast[.float32]() / sigma_q_t
                ).cast[bf16_type]()

            # Also update the BF16 reference Q_rope (used by mha_gpu_naive)
            var ref_rope_start = row_idx * LOGICAL_DEPTH + V_DEPTH
            for d in range(ROPE_DIM):
                q_ref_bf16[ref_rope_start + d] = (
                    q_ref_bf16[ref_rope_start + d].cast[.float32]() / sigma_q_t
                ).cast[bf16_type]()

    # -------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -------------------------------------------------------------------
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -------------------------------------------------------------------
    # Step 4: Output tensor [total_q_tokens, num_heads, V_DEPTH=512] BF16
    # -------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = ctx.enqueue_create_host_buffer[bf16_type](out_size)

    # -------------------------------------------------------------------
    # Step 5: Copy to device
    # -------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[fp8_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var scales_device = ctx.enqueue_create_buffer[.float32](scales_elems)
    ctx.enqueue_copy(scales_device, scales_host)

    var q_scales_device = ctx.enqueue_create_buffer[.float32](total_q_tokens)
    ctx.enqueue_copy(q_scales_device, q_scales_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[fp8_type](q_size)
    ctx.enqueue_copy(q_device, q_host_fp8)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[bf16_type](out_size)

    ctx.synchronize()

    # -------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection
    # -------------------------------------------------------------------
    var blocks_lt = LayoutTensor[fp8_type, Layout.row_major[6]()](
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

    # Build PagedKVCacheCollection WITH per-token scales.
    # quantization_granularity_ = PHYSICAL_DIM gives head_dim_gran = 1
    # (one scale per token), matching the per-token scale layout.
    var kv_collection = PagedKVCacheCollection[
        fp8_type,
        kv_params,
        PAGE_SIZE,
        scale_dtype_=DType.float32,
        quantization_granularity_=PHYSICAL_DIM,
    ](
        LayoutTensor[fp8_type, Layout.row_major[6]()](
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
        # Pass the scales tensor
        LayoutTensor[.float32, Layout.row_major[6]()](
            scales_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                scales_lt.runtime_layout.shape.value,
                scales_lt.runtime_layout.stride.value,
            ),
        ),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # Q: [total_q_tokens, num_heads, 640] float8_e4m3fn
    var q_tt = TileTensor(
        q_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[PHYSICAL_DIM])),
    )

    # Output: [total_q_tokens, num_heads, V_DEPTH=512] bfloat16
    var out_tt = TileTensor(
        out_device,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged layout
    var row_offsets_tt = TileTensor(
        row_offsets_device,
        row_major(batch_size + 1),
    )

    # q_scale_ptr: reinterpret as Pointer with MutAnyOrigin
    var q_scale_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        q_scales_device.unsafe_ptr()
    )

    # -------------------------------------------------------------------
    # Step 7: Call the full production dispatch path with scales
    # -------------------------------------------------------------------
    print("  Launching MLA decode kernel (full dispatch path, WITH scales)...")

    # Pre-compute scalar dispatch args (batch_size, max_cache_len, q_max_seq_len).
    var mla_args2 = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt2 = mla_args2.gpu_tile_tensor()
    flare_mla_decoding[
        rank=3,
        config=MHAConfig[fp8_type](num_heads, LOGICAL_DEPTH),
        ragged=True,
        per_token_scale_rope_aware=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf=scalar_args_buf_tt2,
        q_max_seq_len=q_max_seq_len,
        q_scale_ptr=q_scale_ptr,
    )

    ctx.synchronize()
    _ = mla_args2^
    print("  Kernel completed.")

    # -------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive per batch
    #
    # Scale Domain Alignment (SnapMLA Eq. 6) has been applied:
    #   - K_rope_aligned[t] = K_rope[t] / sigma_KV[t]  (Step 1d)
    #   - Q_rope_aligned[q] = Q_rope[q] / sigma_Q[q]   (Step 2b)
    # So the reference applies sigma uniformly to all dims:
    #   - Q_ref = sigma_Q * Q_dequant  (all 576 dims including aligned rope)
    #   - K_ref[t] = sigma_KV[t] * K_dequant[t]  (all 576 dims)
    # For content: sigma_Q * sigma_KV[t] * Q_nope_fp8 · K_nope_fp8[t]
    # For rope:    sigma_Q * sigma_KV[t] * (Q_rope/sigma_Q) · (K_rope/sigma_KV[t])
    #            = Q_rope · K_rope  (sigmas cancel — exact, not approximate)
    # This matches the kernel's uniform scaling, now mathematically exact.
    # -------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print(
        "  Computing GPU naive reference per batch (with scales) and"
        " comparing..."
    )

    var rtol = 5e-2
    var atol = 3e-1
    var total_checked = 0
    var max_abs_err = Float64(0)

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var ref_num_keys = cache_len + q_max_seq_len
        var sigma_q = q_scales_host[b]

        # Extract contiguous BF16 K for this batch from paged blocks,
        # applying sigma_KV[t] per token.
        # K shape: [ref_num_keys, KV_NUM_HEADS, LOGICAL_DEPTH=576]
        var k_b_size = ref_num_keys * KV_NUM_HEADS * LOGICAL_DEPTH
        var k_b_host = ctx.enqueue_create_host_buffer[bf16_type](k_b_size)

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            # Get sigma_KV[t] for this token from the scales host array.
            var scale_offset = (
                physical_page * scale_page_stride
                + tok_in_page * kv_params.num_heads * head_dim_gran
            )
            var sigma_kv_t = scales_host[scale_offset]

            # Extract BF16 data (dequant content + copy rope)
            var dst_offset = tok * KV_NUM_HEADS * LOGICAL_DEPTH
            extract_bf16_kv_from_block(
                blocks_host.unsafe_ptr(),
                k_b_host.unsafe_ptr() + dst_offset,
                physical_page,
                tok_in_page,
                kv_dim2,
                _page_stride,
            )

            # Apply sigma_KV[t] to all dims of this token
            for kh in range(KV_NUM_HEADS):
                for d in range(LOGICAL_DEPTH):
                    var idx = dst_offset + kh * LOGICAL_DEPTH + d
                    k_b_host[idx] = (
                        k_b_host[idx].cast[.float32]() * sigma_kv_t
                    ).cast[bf16_type]()

        # Q for this batch: BF16 [1, 1, num_heads, LOGICAL_DEPTH=576]
        # Apply sigma_Q to Q
        var q_b_size = 1 * num_heads * LOGICAL_DEPTH
        var q_b_host = ctx.enqueue_create_host_buffer[bf16_type](q_b_size)
        for i in range(q_b_size):
            q_b_host[i] = (
                q_ref_bf16[b * num_heads * LOGICAL_DEPTH + i].cast[
                    DType.float32
                ]()
                * sigma_q
            ).cast[bf16_type]()

        # Reference output: [1, 1, num_heads, LOGICAL_DEPTH=576] (full depth)
        var ref_b_size = 1 * num_heads * LOGICAL_DEPTH
        var ref_b_host = ctx.enqueue_create_host_buffer[bf16_type](ref_b_size)

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[bf16_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[bf16_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[bf16_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors for mha_gpu_naive reference
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[LOGICAL_DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[LOGICAL_DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[LOGICAL_DEPTH])),
        )

        # mha_gpu_naive: K used as both K and V (MLA: V = K[:,:,:512])
        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            NullMask(),
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            1,  # seq_len
            ref_num_keys,
            num_heads,
            LOGICAL_DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims per head
        var out_offset = b * num_heads * V_DEPTH
        var nan_count = 0
        var first_nan_h = -1
        var first_nan_d = -1
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host[d + LOGICAL_DEPTH * h].cast[
                    DType.float64
                ]()
                var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                    DType.float64
                ]()
                if actual != actual:  # NaN check
                    nan_count += 1
                    if first_nan_h == -1:
                        first_nan_h = h
                        first_nan_d = d
                        print(
                            "  NaN at batch",
                            b,
                            "head",
                            h,
                            "depth",
                            d,
                            "expect",
                            expect,
                        )
                    continue
                var abs_err = abs(actual - expect)
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if abs_err > 1e-1:
                    print(b, h, d, actual, expect)
        if nan_count > 0:
            print(
                "  TOTAL NaNs for batch",
                b,
                ":",
                nan_count,
                "out of",
                num_heads * V_DEPTH,
            )
            print(
                "  First NaN at h=",
                first_nan_h,
                "d=",
                first_nan_d,
            )
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var expect = ref_b_host[d + LOGICAL_DEPTH * h].cast[
                    DType.float64
                ]()
                var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                    DType.float64
                ]()
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

        total_checked += num_heads * V_DEPTH

        # Cleanup per-batch buffers
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

    # Cleanup

    _ = blocks_device
    _ = scales_device
    _ = q_scales_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device


def main() raises:
    seed(42)
    print("Starting test_mla_decode_qkv_fp8_per_token_scale_rope_aware...")
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            print("=" * 72)
            print(
                "MLA Decode FP8 Per-Tensor Rope-Aware Test (B200) -"
                " Full Dispatch Path"
            )
            print("=" * 72)
            print()

            # -----------------------------------------------------------
            # Group 1: Small caches (16 heads)
            # Tests basic functionality with short cache lengths.
            # -----------------------------------------------------------
            print("--- Group 1: Small caches (16 heads) ---")

            var cl1: List[Int] = [30]
            run_test[16]("small_30", cl1, ctx)

            var cl2: List[Int] = [50]
            run_test[16]("small_50", cl2, ctx)

            var cl3: List[Int] = [80]
            run_test[16]("small_80", cl3, ctx)

            var cl4: List[Int] = [100]
            run_test[16]("small_100", cl4, ctx)

            var cl5: List[Int] = [256]
            run_test[16]("small_256", cl5, ctx)

            # -----------------------------------------------------------
            # Group 2: Variable cache lengths (16 heads)
            # Tests tile boundaries and split-K partitioning.
            # -----------------------------------------------------------
            print("--- Group 2: Variable cache lengths (16 heads) ---")

            var cl6: List[Int] = [640]
            run_test[16]("medium_640", cl6, ctx)

            var cl7: List[Int] = [1024]
            run_test[16]("medium_1024", cl7, ctx)

            var cl8: List[Int] = [2048]
            run_test[16]("medium_2048", cl8, ctx)

            # -----------------------------------------------------------
            # Group 3: Large caches (16 heads)
            # Exercises split-K with many tiles.
            # -----------------------------------------------------------
            print("--- Group 3: Large caches (16 heads) ---")

            var cl9: List[Int] = [4096]
            run_test[16]("large_4096", cl9, ctx)

            var cl10: List[Int] = [8192]
            run_test[16]("large_8192", cl10, ctx)

            var cl11: List[Int] = [9600]
            run_test[16]("large_9600", cl11, ctx)

            var cl12: List[Int] = [16384]
            run_test[16]("large_16384", cl12, ctx)

            var cl13: List[Int] = [32768]
            run_test[16]("large_32768", cl13, ctx)

            # -----------------------------------------------------------
            # Group 4: Higher head counts (64, 128)
            # -----------------------------------------------------------
            print("--- Group 4: Higher head counts ---")

            var cl14: List[Int] = [256]
            run_test[64]("64heads_256", cl14, ctx)

            var cl15: List[Int] = [4096]
            run_test[64]("64heads_4096", cl15, ctx)

            var cl16: List[Int] = [16384]
            run_test[64]("64heads_16384", cl16, ctx)

            var cl17: List[Int] = [256]
            run_test[128]("128heads_256", cl17, ctx)

            var cl18: List[Int] = [1024]
            run_test[128]("128heads_1024", cl18, ctx)

            var cl19: List[Int] = [8192]
            run_test[128]("128heads_8192", cl19, ctx)

            var cl20: List[Int] = [32768]
            run_test[128]("128heads_32768", cl20, ctx)

            # -----------------------------------------------------------
            # Group 5: Multi-batch tests
            # Tests split-K with multiple batches.
            # -----------------------------------------------------------
            print("--- Group 5: Multi-batch tests ---")

            var cl21: List[Int] = [4096, 4096]
            run_test[16]("bs2_4096", cl21, ctx)

            var cl22: List[Int] = [1024, 1024, 1024, 1024]
            run_test[16]("bs4_1024", cl22, ctx)

            var cl23: List[Int] = [
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
            ]
            run_test[16]("bs8_4096", cl23, ctx)

            var cl24: List[Int] = [4096, 4096]
            run_test[128]("128heads_bs2_4096", cl24, ctx)

            var cl25: List[Int] = [1024, 1024, 1024, 1024]
            run_test[128]("128heads_bs4_1024", cl25, ctx)

            # -----------------------------------------------------------
            # Group 6: Variable cache lengths per batch
            # Tests split-K with different cache lengths per batch
            # (the main benefit of the split-K path).
            # -----------------------------------------------------------
            print("--- Group 6: Variable cache lengths per batch ---")

            var cl26: List[Int] = [30, 256, 640, 1024]
            run_test[16]("variable_short", cl26, ctx)

            var cl27: List[Int] = [30, 1024, 8192, 32768]
            run_test[16]("variable_extreme", cl27, ctx)

            var cl28: List[Int] = [30, 128, 256, 512, 1024, 4096, 16384, 32768]
            run_test[16]("variable_mixed_8batch", cl28, ctx)

            var cl29: List[Int] = [256, 4096, 16384, 32768]
            run_test[128]("128heads_variable", cl29, ctx)

            # -----------------------------------------------------------
            # Group 7: Production-representative configs
            # -----------------------------------------------------------
            print("--- Group 7: Production-representative configs ---")

            var cl30: List[Int] = [
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
            ]
            run_test[16]("bs16_4096", cl30, ctx)

            var cl31: List[Int] = [
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
                4096,
            ]
            run_test[128]("128heads_bs8_4096", cl31, ctx)

            # -----------------------------------------------------------
            # Group 8: Very small cache_len
            # -----------------------------------------------------------
            print("--- Group 8: Very small cache_len ---")

            var cl32: List[Int] = [64]
            run_test[16]("tiny_64", cl32, ctx)

            var cl33: List[Int] = [128]
            run_test[16]("tiny_128", cl33, ctx)

            var cl34: List[Int] = [512]
            run_test[16]("tiny_512", cl34, ctx)

            # -----------------------------------------------------------
            # Group 9: Non-power-of-2 cache_len
            # -----------------------------------------------------------
            print("--- Group 9: Non-power-of-2 cache_len ---")

            var cl35: List[Int] = [7777]
            run_test[16]("nonpow2_7777", cl35, ctx)

            var cl36: List[Int] = [50000]
            run_test[16]("nonpow2_50000", cl36, ctx)

            # -----------------------------------------------------------
            # Group 10: Large cache_len (>64K)
            # -----------------------------------------------------------
            print("--- Group 10: Large cache_len ---")

            var cl37: List[Int] = [65536]
            run_test[16]("huge_65536", cl37, ctx)

            var cl38: List[Int] = [163840]
            run_test[16]("huge_163840", cl38, ctx)

            # -----------------------------------------------------------
            # Group 11: Full DeepSeek 128-head production
            # -----------------------------------------------------------
            print("--- Group 11: Full DeepSeek 128-head production ---")

            var cl39: List[Int] = [
                1024,
                1024,
                1024,
                1024,
                1024,
                1024,
                1024,
                1024,
            ]
            run_test[128]("128heads_bs8_1024", cl39, ctx)

            var cl40: List[Int] = [8192, 8192, 8192, 8192]
            run_test[128]("128heads_bs4_8192", cl40, ctx)

            # ===========================================================
            # NON-TRIVIAL SCALES TESTS
            # These tests pass sigma_KV != 1.0 and sigma_Q != 1.0 to
            # validate the per-token scale code paths are applied
            # correctly (QK dequant + PV pre-fuse + softmax scaling).
            # ===========================================================

            # -----------------------------------------------------------
            # Group S1: Small caches with scales (16 heads)
            # Basic validation that non-trivial scales are applied.
            # -----------------------------------------------------------
            print(
                "--- Group S1: Small caches with non-trivial"
                " scales (16 heads) ---"
            )

            var scl1: List[Int] = [30]
            run_test_with_scales[16]("scales_small_30", scl1, ctx)

            var scl2: List[Int] = [100]
            run_test_with_scales[16]("scales_small_100", scl2, ctx)

            var scl3: List[Int] = [256]
            run_test_with_scales[16]("scales_small_256", scl3, ctx)

            # -----------------------------------------------------------
            # Group S2: Medium caches with scales (16 heads)
            # Tests scale application across multiple split-K tiles.
            # -----------------------------------------------------------
            print(
                "--- Group S2: Medium caches with non-trivial"
                " scales (16 heads) ---"
            )

            var scl4: List[Int] = [640]
            run_test_with_scales[16]("scales_medium_640", scl4, ctx)

            var scl5: List[Int] = [1024]
            run_test_with_scales[16]("scales_medium_1024", scl5, ctx)

            var scl6: List[Int] = [4096]
            run_test_with_scales[16]("scales_medium_4096", scl6, ctx)

            # -----------------------------------------------------------
            # Group S3: Large caches with scales (16 heads)
            # Exercises scale loading across many split-K partitions.
            # -----------------------------------------------------------
            print(
                "--- Group S3: Large caches with non-trivial"
                " scales (16 heads) ---"
            )

            var scl7: List[Int] = [8192]
            run_test_with_scales[16]("scales_large_8192", scl7, ctx)

            var scl8: List[Int] = [32768]
            run_test_with_scales[16]("scales_large_32768", scl8, ctx)

            # -----------------------------------------------------------
            # Group S4: Higher head counts with scales (64, 128)
            # -----------------------------------------------------------
            print(
                "--- Group S4: Higher head counts with non-trivial scales ---"
            )

            var scl9: List[Int] = [256]
            run_test_with_scales[64]("scales_64heads_256", scl9, ctx)

            var scl10: List[Int] = [4096]
            run_test_with_scales[64]("scales_64heads_4096", scl10, ctx)

            var scl11: List[Int] = [256]
            run_test_with_scales[128]("scales_128heads_256", scl11, ctx)

            var scl12: List[Int] = [1024]
            run_test_with_scales[128]("scales_128heads_1024", scl12, ctx)

            var scl13: List[Int] = [8192]
            run_test_with_scales[128]("scales_128heads_8192", scl13, ctx)

            # -----------------------------------------------------------
            # Group S5: Multi-batch with scales
            # Tests per-batch sigma_Q indexing and per-token sigma_KV
            # across multiple batch entries.
            # -----------------------------------------------------------
            print("--- Group S5: Multi-batch with non-trivial scales ---")

            var scl14: List[Int] = [1024, 1024]
            run_test_with_scales[16]("scales_bs2_1024", scl14, ctx)

            var scl15: List[Int] = [1024, 1024, 1024, 1024]
            run_test_with_scales[16]("scales_bs4_1024", scl15, ctx)

            var scl16: List[Int] = [4096, 4096]
            run_test_with_scales[128]("scales_128heads_bs2_4096", scl16, ctx)

            var scl17: List[Int] = [1024, 1024, 1024, 1024]
            run_test_with_scales[128]("scales_128heads_bs4_1024", scl17, ctx)

            # -----------------------------------------------------------
            # Group S6: Variable cache lengths per batch with scales
            # Tests scale indexing correctness with different page
            # counts per batch entry.
            # -----------------------------------------------------------
            print(
                "--- Group S6: Variable cache lengths with"
                " non-trivial scales ---"
            )

            var scl18: List[Int] = [30, 256, 640, 1024]
            run_test_with_scales[16]("scales_variable_short", scl18, ctx)

            var scl19: List[Int] = [30, 1024, 8192, 32768]
            run_test_with_scales[16]("scales_variable_extreme", scl19, ctx)

            var scl20: List[Int] = [256, 4096, 16384]
            run_test_with_scales[128]("scales_128heads_variable", scl20, ctx)

            print("=" * 72)
            print("ALL TESTS PASSED")
            print("=" * 72)
        else:
            print("Skipping: requires B200 GPU")
