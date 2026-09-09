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

"""Test MLA decode with paged KV cache and variable sequence lengths.

This test exercises the splitK ragged layout path that was fixed:
  Bug A: o_accum_split layout mismatch (ragged vs padded)
  Bug B: LSE stride mismatch (per-batch seq_len vs q_max_seq_len)
  Bug C: Combine kernel output indexing (padded vs ragged)

The key stress scenarios:
  - Variable KV cache lengths per batch (30 to 32768 tokens), causing
    different num_partitions per batch and many empty splits
  - Extreme disparity in cache lengths within the same batch
"""

from std.math import align_up, ceildiv
from std.random import randn, seed
from std.sys import (
    argv,
    get_defined_int,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

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
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.testing import assert_almost_equal
from max.gpu.host.info import B200, _is_sm10x_gpu
from std.utils.index import Index, IndexList


# ===-----------------------------------------------------------------------===#
# Test parameters matching DeepSeek-V2-Lite
# ===-----------------------------------------------------------------------===#

comptime DEPTH = 576  # Q/K head dimension
comptime V_DEPTH = 512  # Output head dimension (depth - 64)
comptime PAGE_SIZE = get_defined_int["page_size", 128]()
comptime NUM_LAYERS = 1  # Single layer for testing
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# ===-----------------------------------------------------------------------===#
# Core test: paged KV cache with variable lengths + numerical verification
# ===-----------------------------------------------------------------------===#


def run_test_paged_variable[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
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
        # Match production: allocate pages for cache_len + new token
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

    # Generate random data in bf16 then cast to kv_type.  This avoids
    # issues with randn producing poorly-distributed values for float8
    # types.  For bf16 kv_type the cast is a no-op.
    # Use std=0.5 to keep QK dot products moderate and softmax
    # numerically stable across all cache lengths.
    var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    # Zero out the unused tail slots in each page so that if the kernel
    # accidentally reads beyond a batch's cache_len, the attention weights
    # on those tokens are negligible (exp of Q*0 is small relative to real data).
    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Match production: pages cover cache_len + new token
    # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
    # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
    var max_pages_per_batch = align_up(
        ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond num_keys).
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            # Zero out tokens [valid_toks, PAGE_SIZE) in this page
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -----------------------------------------------------------------------
    # Step 4: Output tensor
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[q_type](0))

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

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

    # First create mutable LayoutTensors from device buffers
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    # Build PagedKVCacheCollection using the pattern from existing tests:
    # wrap .ptr into LayoutTensors with appropriate origins
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA decode kernel...")

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

    print("  Computing GPU naive reference per batch and comparing...")

    # Element-wise comparison: same tolerances as test_mla_decode_kv_fp8.
    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
    var total_checked = 0
    var max_abs_err = Float64(0)

    # Run mha_gpu_naive per batch with correct num_keys to avoid
    # zero-padding effects from shorter caches.
    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        # num_keys includes the new token (production: cache_len + seq_len)
        var ref_num_keys = cache_len + q_max_seq_len

        # Extract contiguous K for this batch from paged blocks
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + tok_in_page * kv_params.num_heads * kv_params.head_size
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst_offset + d] = blocks_host[src_offset + d]

        # Q for this batch: [1, 1, num_heads, depth]
        var q_b_size = 1 * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
        for i in range(q_b_size):
            q_b_host[i] = q_host[b * num_heads * DEPTH + i]

        # Reference output: [1, 1, num_heads, depth] (full depth)
        var ref_b_size = 1 * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors for mha_gpu_naive reference
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
        )

        # Run mha_gpu_naive: batch_size=1, num_keys=ref_num_keys
        # K passed as both K and V (MLA: V = K[:,:,:512])
        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            NullMask(),
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            1,  # seq_len
            ref_num_keys,  # num_keys (cache_len + new token)
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

        # Cleanup per-batch buffers
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_host^
    _ = blocks_bf16^


# ===-----------------------------------------------------------------------===#
# Core test: paged KV cache with variable lengths AND q_max_seq_len > 1
# ===-----------------------------------------------------------------------===#


def run_test_paged_variable_multiq[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    q_max_seq_len: Int,
    ctx: DeviceContext,
) raises:
    """Test MLA decode with paged KV cache, variable cache lengths, AND
    q_max_seq_len > 1 (multiple query tokens per batch entry).

    This exercises the ragged layout path where block_idx.y iterates over
    query tokens within a batch entry, testing that:
    - Q row offsets are computed correctly for multi-token decode
    - Output row offsets include the seq_len dimension
    - The combine kernel's o_accum_split layout handles the padded seq dimension
    - FP8 blockwise scale loading is independent of Q seq_len
    """
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
        " q_max_seq_len:",
        q_max_seq_len,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    var total_q_tokens = batch_size * q_max_seq_len

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        # Match production: allocate pages for cache_len + new tokens
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

    var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Match production: pages cover cache_len + new tokens
    # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
    # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
    var max_pages_per_batch = align_up(
        ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond num_keys).
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + q_max_seq_len
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            # Zero out tokens [valid_toks, PAGE_SIZE) in this page
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # Each batch has q_max_seq_len tokens
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(q_max_seq_len)

    # -----------------------------------------------------------------------
    # Step 4: Output tensor
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[q_type](0))

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

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

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA decode kernel...")

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

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    var rtol = 5e-2  # 0.05
    var atol = 3e-1  # 0.3
    var total_checked = 0
    var max_abs_err = Float64(0)

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var ref_num_keys = cache_len + q_max_seq_len

        # Extract contiguous K for this batch from paged blocks
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + tok_in_page * kv_params.num_heads * kv_params.head_size
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst_offset + d] = blocks_host[src_offset + d]

        # Q for this batch: [1, q_max_seq_len, num_heads, depth]
        var q_b_size = q_max_seq_len * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
        var q_batch_offset = b * q_max_seq_len * num_heads * DEPTH
        for i in range(q_b_size):
            q_b_host[i] = q_host[q_batch_offset + i]

        # Reference output: [1, q_max_seq_len, num_heads, depth] (full depth)
        var ref_b_size = q_max_seq_len * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors for mha_gpu_naive reference
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], q_max_seq_len, Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], q_max_seq_len, Idx[num_heads], Idx[DEPTH])),
        )

        # Run mha_gpu_naive: batch_size=1, seq_len=q_max_seq_len
        # K passed as both K and V (MLA: V = K[:,:,:512])
        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            NullMask(),
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            q_max_seq_len,  # seq_len
            ref_num_keys,  # num_keys (cache_len + new tokens)
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims (depth-64) per head, per query token
        # ref layout: [1, q_max_seq_len, num_heads, depth]
        # actual layout: [total_tokens, num_heads, V_DEPTH]
        for s in range(q_max_seq_len):
            var out_offset = (b * q_max_seq_len + s) * num_heads * V_DEPTH
            var ref_s_offset = s * num_heads * DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var expect = ref_b_host[ref_s_offset + d + DEPTH * h].cast[
                        DType.float64
                    ]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                        DType.float64
                    ]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > 1e-1:
                        print(b, s, h, d, actual, expect)
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

            total_checked += num_heads * V_DEPTH

        # Cleanup per-batch buffers
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_host^
    _ = blocks_bf16^


# ===-----------------------------------------------------------------------===#
# Core test: truly ragged Q — each batch has a DIFFERENT number of Q tokens
# ===-----------------------------------------------------------------------===#


def run_test_paged_variable_ragged_q[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    ctx: DeviceContext,
) raises:
    """Test MLA decode with paged KV cache AND truly variable per-batch Q
    sequence lengths (ragged Q).

    Unlike run_test_paged_variable_multiq (where all batches have the same
    q_max_seq_len), this test gives each batch a DIFFERENT number of query
    tokens.  For example:
        batch 0: 1 Q token
        batch 1: 3 Q tokens
        batch 2: 2 Q tokens
        batch 3: 4 Q tokens
        q_max_seq_len = 4  (the max)
        row_offsets = [0, 1, 4, 6, 10]

    The kernel grid launches grid_dim.y = q_max_seq_len CTAs per batch.
    CTAs with block_idx.y >= per-batch seq_len early-exit via the ragged
    guard in OffsetPosition.

    This verifies:
    - OffsetPosition.seq_len is correctly derived from row_offsets per batch
    - The early-exit path writes -inf LSE and zeroed o_accum_split for the
      combine kernel
    - num_keys = cache_length + per-batch seq_len (NOT + q_max_seq_len)
    - Page allocation covers cache_len + seq_lens[i] per batch
    - FP8 blockwise scaling is independent of variable Q lengths
    """
    var batch_size = len(cache_lengths)
    assert (
        len(seq_lens) == batch_size
    ), "cache_lengths and seq_lens must have same length"
    var q_max_seq_len = 0
    var total_q_tokens = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]

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
        " q_max_seq_len:",
        q_max_seq_len,
        " total_q_tokens:",
        total_q_tokens,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " q_seq_len=",
            seq_lens[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        # Each batch needs pages for cache_len + its own seq_len
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

    var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # max_pages_per_batch uses cache_len + seq_len for the max batch
    # We need the lookup table to be wide enough for the batch with the
    # most pages.
    var max_num_keys_any_batch = 0
    for i in range(batch_size):
        var nk = cache_lengths[i] + seq_lens[i]
        if nk > max_num_keys_any_batch:
            max_num_keys_any_batch = nk
    # Pad to multiple of 8 for LUT row stride alignment (SIMD populate).
    var max_pages_per_batch = align_up(
        ceildiv(max_num_keys_any_batch, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond num_keys).
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + seq_lens[bi]
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor (ragged: [total_q_tokens, num_heads, DEPTH])
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets — truly ragged per batch
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    # -----------------------------------------------------------------------
    # Step 4: Output tensor
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[q_type](0))

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

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

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    # max_seq_length = q_max_seq_len (the maximum across all batches)
    # max_cache_length = max_cache_len (the maximum cache length)
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching MLA decode kernel...")

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

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    var rtol = 5e-2
    var atol = 3e-1
    var total_checked = 0
    var max_abs_err = Float64(0)

    # Track where each batch's Q tokens start in the ragged output
    var q_token_offset = 0

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var b_seq_len = seq_lens[b]
        # num_keys for this batch = cache_len + this batch's seq_len
        var ref_num_keys = cache_len + b_seq_len

        # Extract contiguous K for this batch from paged blocks
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[kv_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + seq_lens[bi], PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + tok_in_page * kv_params.num_heads * kv_params.head_size
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst_offset + d] = blocks_host[src_offset + d]

        # Q for this batch: [1, b_seq_len, num_heads, depth]
        var q_b_size = b_seq_len * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[q_type](0))
        var q_batch_start = q_token_offset * num_heads * DEPTH
        for i in range(q_b_size):
            q_b_host[i] = q_host[q_batch_start + i]

        # Reference output: [1, b_seq_len, num_heads, depth] (full depth)
        var ref_b_size = b_seq_len * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[q_type](0))

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[kv_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[q_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[q_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors for mha_gpu_naive reference
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )

        # Run mha_gpu_naive: batch_size=1, seq_len=b_seq_len
        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            NullMask(),
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            b_seq_len,  # this batch's seq_len
            ref_num_keys,  # cache_len + this batch's seq_len
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH dims per head, per query token
        # ref layout: [1, b_seq_len, num_heads, depth]
        # actual layout: [total_tokens, num_heads, V_DEPTH]  (ragged)
        for s in range(b_seq_len):
            var out_offset = (q_token_offset + s) * num_heads * V_DEPTH
            var ref_s_offset = s * num_heads * DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var expect = ref_b_host[ref_s_offset + d + DEPTH * h].cast[
                        DType.float64
                    ]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                        DType.float64
                    ]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > 1e-1:
                        print(
                            "batch",
                            b,
                            "tok",
                            s,
                            "head",
                            h,
                            "dim",
                            d,
                            "actual",
                            actual,
                            "expect",
                            expect,
                        )
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

            total_checked += num_heads * V_DEPTH

        q_token_offset += b_seq_len

        # Cleanup per-batch buffers
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_host^
    _ = blocks_bf16^


# ===-----------------------------------------------------------------------===#
# Benchmark: paged KV cache with variable lengths (no numerical verification)
# ===-----------------------------------------------------------------------===#


def run_bench_paged_variable[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    var batch_size = len(cache_lengths)
    print(
        "bench:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " q_type:",
        q_type,
        " kv_type:",
        kv_type,
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
        # Match production: allocate pages for cache_len + new token
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    comptime scale = Float32(0.125)

    # Step 1: Paged KV cache on host
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_bf16 = List(length=block_elems, fill=Scalar[q_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Match production: pages cover cache_len + new token
    # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
    # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
    var max_pages_per_batch = align_up(
        ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in each page (tokens beyond num_keys)
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
                blocks_host[base + z] = 0
            cur_page += 1

    # Step 2: Q tensor
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # Step 3: input_row_offsets
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # Step 4: Output tensor
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[q_type](0))

    # Step 5: Copy to device
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
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

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_host^
    _ = blocks_bf16^


# ===-----------------------------------------------------------------------===#
# Core test: native FP8 paged path (Q=FP8, KV=FP8, output=BF16)
# ===-----------------------------------------------------------------------===#


def run_test_paged_variable_native_fp8[
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    """Test native FP8 MLA decode through the paged path.

    In the native FP8 path:
    - Q is float8_e4m3fn (sent to the kernel as FP8 via TMA)
    - KV cache is float8_e4m3fn (tensorwise, scale_block_size=0)
    - Output is bfloat16

    This exercises the 3-WG native FP8 kernel (MLA_SM100_Decode_QKV_FP8)
    with split-K + PDL through the paged KV cache path.

    For the GPU naive reference, we use BF16 Q and dequantized BF16 K
    since mha_gpu_naive does not support FP8 inputs.
    """
    comptime q_type = DType.float8_e4m3fn
    comptime kv_type = DType.float8_e4m3fn
    comptime ref_type = DType.bfloat16  # Reference/output type

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
        " output_type:",
        ref_type,
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
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host (FP8)
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))

    # Generate KV data in bf16 first, then cast to FP8.
    # Keep bf16 copy for the reference path.
    var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()
    # Note: blocks_bf16 is kept alive for reference computation below.

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
    # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
    var max_pages_per_batch = align_up(
        ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in both FP8 and BF16 copies
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
                blocks_host[base + z] = 0
                blocks_bf16[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor — generate in BF16, keep copy, cast to FP8
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH

    # BF16 Q for reference
    var q_bf16_host = List(length=q_size, fill=Scalar[ref_type](0))
    randn(
        q_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # FP8 Q for kernel
    var q_fp8_host = List(length=q_size, fill=Scalar[q_type](0))
    for i in range(q_size):
        q_fp8_host[i] = q_bf16_host[i].cast[q_type]()

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets (batch_size + 1 elements)
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # -----------------------------------------------------------------------
    # Step 4: Output tensor (BF16, not FP8)
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[ref_type](0))

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    # FP8 Q to device
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_fp8_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    # Output is BF16
    var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # Q is FP8, output is BF16
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
    # Step 7: Pre-compute scalar args and call the kernel (Q=FP8, KV=FP8, output=BF16)
    # -----------------------------------------------------------------------
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()
    print("  Launching native FP8 MLA decode kernel...")

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

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference
    # Uses BF16 Q and dequantized BF16 K for the reference.
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    # FP8 has larger quantization error, so use wider tolerances.
    var rtol = 8e-2
    var atol = 5e-1
    var total_checked = 0
    var max_abs_err = Float64(0)

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var ref_num_keys = cache_len + q_max_seq_len

        # Extract contiguous BF16 K for this batch from bf16 paged blocks
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[ref_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + q_max_seq_len, PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + tok_in_page * kv_params.num_heads * kv_params.head_size
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                # Use BF16 (dequantized) version for reference
                k_b_host[dst_offset + d] = blocks_bf16[src_offset + d]

        # BF16 Q for this batch: [1, 1, num_heads, depth]
        var q_b_size = 1 * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[ref_type](0))
        for i in range(q_b_size):
            q_b_host[i] = q_bf16_host[b * num_heads * DEPTH + i]

        # Reference output: [1, 1, num_heads, depth] (full depth, BF16)
        var ref_b_size = 1 * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[ref_type](0))

        # Copy to device
        var k_b_device = ctx.enqueue_create_buffer[ref_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[ref_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[ref_type](ref_b_size)
        ctx.synchronize()

        # Build 4D TileTensors (all BF16 for reference)
        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], Idx[1], Idx[num_heads], Idx[DEPTH])),
        )

        # Run mha_gpu_naive with BF16 inputs
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
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        # Compare first V_DEPTH=512 dims per head
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
                if abs_err > 2e-1:
                    print(b, h, d, actual, expect)
                assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

        total_checked += num_heads * V_DEPTH

        # Cleanup per-batch buffers
        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_bf16^
    _ = blocks_host^
    _ = out_host^
    _ = q_bf16_host^
    _ = q_fp8_host^


# ===-----------------------------------------------------------------------===#
# Core test: native FP8 paged path with truly ragged Q (per-batch seq_len)
# ===-----------------------------------------------------------------------===#


def run_test_paged_variable_ragged_q_native_fp8[
    mask_t: MHAMask,
    //,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    mask: mask_t,
    ctx: DeviceContext,
) raises:
    """Native FP8 (Q=FP8, KV=FP8, output=BF16) MLA decode through the paged KV
    cache with per-batch variable Q sequence lengths (ragged Q).

    This is the AMD MTP fold exercised through the production paged + ragged
    layout path (the AMD fold is FP8-only and requires num_heads <= 16). Uniform
    `seq_lens` give a uniform multi-Q fold; differing `seq_lens` give the
    variable-draft-length case, which is supported: the comptime `q_seq_len`
    (== q_max_seq_len) is only the padding ceiling, while the runtime
    per-sequence length drives the SRD valid-row clamp, the per-token causal
    offset, the split-K stat writer, and the ragged reducer, so short sequences'
    pad rows are dropped. Pass `CausalMask()` to exercise that per-token causal
    offset (the discriminating case for variable Q); `NullMask()` covers the
    unmasked reduction.

    `mask` is applied to both the kernel and the per-batch GPU-naive reference,
    which uses BF16 Q and dequantized BF16 K with `num_keys = cache_len + this
    batch's seq_len` — so `mha_gpu_naive`'s `score_row = y + cache_len` matches
    the kernel's per-token `num_keys - seq_len + token`.
    """
    comptime q_type = DType.float8_e4m3fn
    comptime kv_type = DType.float8_e4m3fn
    comptime ref_type = DType.bfloat16

    var batch_size = len(cache_lengths)
    assert (
        len(seq_lens) == batch_size
    ), "cache_lengths and seq_lens must have same length"
    var q_max_seq_len = 0
    var total_q_tokens = 0
    for i in range(batch_size):
        if seq_lens[i] > q_max_seq_len:
            q_max_seq_len = seq_lens[i]
        total_q_tokens += seq_lens[i]

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
        " q_max_seq_len:",
        q_max_seq_len,
        " total_q_tokens:",
        total_q_tokens,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " q_seq_len=",
            seq_lens[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    for i in range(batch_size):
        if cache_lengths[i] > max_cache_len:
            max_cache_len = cache_lengths[i]
        total_pages += ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)

    comptime scale = Float32(0.125)
    comptime group = num_heads

    # -----------------------------------------------------------------------
    # Step 1: Create the paged KV cache on host (FP8 + BF16 reference copy)
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var max_num_keys_any_batch = 0
    for i in range(batch_size):
        var nk = cache_lengths[i] + seq_lens[i]
        if nk > max_num_keys_any_batch:
            max_num_keys_any_batch = nk
    var max_pages_per_batch = align_up(
        ceildiv(max_num_keys_any_batch, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + seq_lens[i], PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

    # Zero out tail slots in both FP8 and BF16 copies.
    var cur_page = 0
    for bi in range(batch_size):
        var num_keys_i = cache_lengths[bi] + seq_lens[bi]
        var num_pages_i = ceildiv(num_keys_i, PAGE_SIZE)
        for pg in range(num_pages_i):
            var valid_toks = num_keys_i - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            var base = cur_page * _page_stride + valid_toks * _tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * _tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = 0
                blocks_bf16[base + z] = 0
            cur_page += 1

    # -----------------------------------------------------------------------
    # Step 2: Q tensor — generate BF16, keep copy, cast to FP8
    # -----------------------------------------------------------------------
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_bf16_host = List(length=q_size, fill=Scalar[ref_type](0))
    randn(q_bf16_host, mean=0.0, standard_deviation=0.5)
    var q_fp8_host = List(length=q_size, fill=Scalar[q_type](0))
    for i in range(q_size):
        q_fp8_host[i] = q_bf16_host[i].cast[q_type]()

    # -----------------------------------------------------------------------
    # Step 3: input_row_offsets — truly ragged per batch
    # -----------------------------------------------------------------------
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(seq_lens[i])

    # -----------------------------------------------------------------------
    # Step 4: Output tensor (BF16)
    # -----------------------------------------------------------------------
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[ref_type](0))

    # -----------------------------------------------------------------------
    # Step 5: Copy everything to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_fp8_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Step 6: Build LayoutTensors and PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
        ctx,
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    print("  Launching native FP8 MLA decode kernel (ragged Q)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, DEPTH),
        ragged=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        mask,
        row_offsets_tt,
        scale,
        ctx,
        scalar_args_buf_tt,
        q_max_seq_len=q_max_seq_len,
    )

    ctx.synchronize()
    print("  Kernel completed successfully (no crash).")

    # -----------------------------------------------------------------------
    # Step 8: Numerical verification using mha_gpu_naive reference (BF16)
    # -----------------------------------------------------------------------
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    print("  Computing GPU naive reference per batch and comparing...")

    var rtol = 8e-2
    var atol = 5e-1
    var total_checked = 0
    var max_abs_err = Float64(0)

    var q_token_offset = 0

    for b in range(batch_size):
        var cache_len = cache_lengths[b]
        var b_seq_len = seq_lens[b]
        var ref_num_keys = cache_len + b_seq_len

        # Extract contiguous BF16 K for this batch from bf16 paged blocks.
        var k_b_size = ref_num_keys * KV_NUM_HEADS * DEPTH
        var k_b_host = List(length=k_b_size, fill=Scalar[ref_type](0))

        var page_base_b = 0
        for bi in range(b):
            page_base_b += ceildiv(cache_lengths[bi] + seq_lens[bi], PAGE_SIZE)

        for tok in range(ref_num_keys):
            var page_idx = tok // PAGE_SIZE
            var tok_in_page = tok % PAGE_SIZE
            var physical_page = page_base_b + page_idx

            var src_offset = (
                physical_page
                * kv_dim2
                * NUM_LAYERS
                * PAGE_SIZE
                * kv_params.num_heads
                * kv_params.head_size
                + tok_in_page * kv_params.num_heads * kv_params.head_size
            )
            var dst_offset = tok * KV_NUM_HEADS * DEPTH
            for d in range(KV_NUM_HEADS * DEPTH):
                k_b_host[dst_offset + d] = blocks_bf16[src_offset + d]

        # BF16 Q for this batch: [1, b_seq_len, num_heads, depth]
        var q_b_size = b_seq_len * num_heads * DEPTH
        var q_b_host = List(length=q_b_size, fill=Scalar[ref_type](0))
        var q_batch_start = q_token_offset * num_heads * DEPTH
        for i in range(q_b_size):
            q_b_host[i] = q_bf16_host[q_batch_start + i]

        var ref_b_size = b_seq_len * num_heads * DEPTH
        var ref_b_host = List(length=ref_b_size, fill=Scalar[ref_type](0))

        var k_b_device = ctx.enqueue_create_buffer[ref_type](k_b_size)
        ctx.enqueue_copy(k_b_device, k_b_host)

        var q_b_device = ctx.enqueue_create_buffer[ref_type](q_b_size)
        ctx.enqueue_copy(q_b_device, q_b_host)

        var ref_b_device = ctx.enqueue_create_buffer[ref_type](ref_b_size)
        ctx.synchronize()

        var q_b_tt = TileTensor(
            q_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )
        var k_b_tt = TileTensor(
            k_b_device,
            row_major(
                (
                    Idx[1],
                    ref_num_keys,
                    Idx[KV_NUM_HEADS],
                    Idx[DEPTH],
                )
            ),
        )
        var ref_b_tt = TileTensor(
            ref_b_device,
            row_major((Idx[1], b_seq_len, Idx[num_heads], Idx[DEPTH])),
        )

        mha_gpu_naive(
            q_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            k_b_tt.to_layout_tensor(),
            mask,
            ref_b_tt.to_layout_tensor(),
            scale,
            1,  # batch_size
            b_seq_len,  # this batch's seq_len
            ref_num_keys,  # cache_len + this batch's seq_len
            num_heads,
            DEPTH,
            group,
            ctx,
        )

        ctx.synchronize()
        ctx.enqueue_copy(ref_b_host, ref_b_device)
        ctx.synchronize()

        for s in range(b_seq_len):
            var out_offset = (q_token_offset + s) * num_heads * V_DEPTH
            var ref_s_offset = s * num_heads * DEPTH
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var expect = ref_b_host[ref_s_offset + d + DEPTH * h].cast[
                        DType.float64
                    ]()
                    var actual = out_host[out_offset + V_DEPTH * h + d].cast[
                        DType.float64
                    ]()
                    var abs_err = abs(actual - expect)
                    if abs_err > max_abs_err:
                        max_abs_err = abs_err
                    if abs_err > 2e-1:
                        print(
                            "batch",
                            b,
                            "tok",
                            s,
                            "head",
                            h,
                            "dim",
                            d,
                            "actual",
                            actual,
                            "expect",
                            expect,
                        )
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

            total_checked += num_heads * V_DEPTH

        q_token_offset += b_seq_len

        _ = k_b_device
        _ = q_b_device
        _ = ref_b_device
        _ = ref_b_host^
        _ = q_b_host^
        _ = k_b_host^

    print(
        "  Verified:",
        total_checked,
        "elements, max_abs_err:",
        max_abs_err,
    )

    print("  PASS:", name, "\n")

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_bf16^
    _ = blocks_host^
    _ = out_host^
    _ = q_bf16_host^
    _ = q_fp8_host^


# ===-----------------------------------------------------------------------===#
# Benchmark: native FP8 paged path (Q=FP8, KV=FP8, output=BF16)
# ===-----------------------------------------------------------------------===#


def run_bench_paged_variable_native_fp8[
    num_heads: Int,
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext,) raises:
    """Benchmark the native FP8 paged path (no numerical verification)."""
    comptime q_type = DType.float8_e4m3fn
    comptime kv_type = DType.float8_e4m3fn
    comptime ref_type = DType.bfloat16

    var batch_size = len(cache_lengths)
    print(
        "bench:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " q_type:",
        q_type,
        " kv_type:",
        kv_type,
        " output_type:",
        ref_type,
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
        total_pages += ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)

    comptime scale = Float32(0.125)

    # Step 1: Paged KV cache on host (FP8)
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=DEPTH, is_mla=True
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_bf16 = List(length=block_elems, fill=Scalar[ref_type](0))
    randn(
        blocks_bf16,
        mean=0.0,
        standard_deviation=0.5,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()

    var _page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    var _tok_stride = kv_params.num_heads * kv_params.head_size

    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Pad to multiple of 8 so the LUT row stride is chunk-aligned for the
    # `PagedKVCache.populate` SIMD path (chunk = min(num_pages, 8)).
    var max_pages_per_batch = align_up(
        ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE), 8
    )
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for i in range(batch_size):
        var num_pages_i = ceildiv(cache_lengths[i] + q_max_seq_len, PAGE_SIZE)
        for p in range(num_pages_i):
            lookup_table_host[i * max_pages_per_batch + p] = UInt32(
                page_offset + p
            )
        page_offset += num_pages_i

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
                blocks_host[base + z] = 0
            cur_page += 1

    # Step 2: Q tensor (FP8)
    var q_size = total_q_tokens * num_heads * DEPTH
    var q_bf16_tmp = List(length=q_size, fill=Scalar[ref_type](0))
    randn(
        q_bf16_tmp,
        mean=0.0,
        standard_deviation=0.5,
    )
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    for i in range(q_size):
        q_host[i] = q_bf16_tmp[i].cast[q_type]()

    # Step 3: input_row_offsets
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    row_offsets_host[0] = UInt32(0)
    for i in range(batch_size):
        row_offsets_host[i + 1] = row_offsets_host[i] + UInt32(1)

    # Step 4: Output tensor (BF16)
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_host = List(length=out_size, fill=Scalar[ref_type](0))

    # Step 5: Copy to device
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)

    var out_device = ctx.enqueue_create_buffer[ref_type](out_size)

    ctx.synchronize()

    # Step 6: Build LayoutTensors and PagedKVCacheCollection
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
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
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=(kv_type == .float8_e4m3fn),
    ](
        batch_size,
        max_cache_len,
        q_max_seq_len,
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

    # Cleanup

    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = row_offsets_device
    _ = out_device
    _ = row_offsets_host^
    _ = lookup_table_host^
    _ = cache_lengths_host^
    _ = blocks_host^
    _ = out_host^
    _ = q_host^
    _ = q_bf16_tmp^
    _ = blocks_bf16^


# ===-----------------------------------------------------------------------===#
# Helper functions to reduce duplication in main()
# ===-----------------------------------------------------------------------===#


def make_uniform(count: Int, value: Int) -> List[Int]:
    """Create a list of `count` identical cache lengths."""
    var result = List[Int]()
    for _ in range(count):
        result.append(value)
    return result^


def run_both_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run correctness test for both bf16 and fp8 KV types."""
    run_test_paged_variable[.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, ctx
    )
    run_test_paged_variable[.bfloat16, DType.float8_e4m3fn, num_heads](
        name + "_fp8", cache_lengths, ctx
    )


def run_uniform_both[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run correctness test with uniform cache lengths for both KV types."""
    run_both_kv_types[num_heads](name, make_uniform(count, value), ctx)


def run_bench_both_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run benchmark for both bf16 and fp8 KV types."""
    run_bench_paged_variable[.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, ctx
    )
    run_bench_paged_variable[.bfloat16, DType.float8_e4m3fn, num_heads](
        name + "_fp8", cache_lengths, ctx
    )


def run_bench_uniform_both[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run benchmark with uniform cache lengths for both KV types."""
    run_bench_both_kv_types[num_heads](name, make_uniform(count, value), ctx)


def run_multiq_both_kv_types[
    num_heads: Int
](
    name: StringLiteral,
    cache_lengths: List[Int],
    q_max_seq_len: Int,
    ctx: DeviceContext,
) raises:
    """Run multi-Q correctness test for both bf16 and fp8 KV types."""
    run_test_paged_variable_multiq[.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, q_max_seq_len, ctx
    )
    run_test_paged_variable_multiq[
        DType.bfloat16, DType.float8_e4m3fn, num_heads
    ](name + "_fp8", cache_lengths, q_max_seq_len, ctx)


def run_multiq_uniform_both[
    num_heads: Int
](
    name: StringLiteral,
    count: Int,
    value: Int,
    q_max_seq_len: Int,
    ctx: DeviceContext,
) raises:
    """Run multi-Q test with uniform cache lengths for both KV types."""
    run_multiq_both_kv_types[num_heads](
        name, make_uniform(count, value), q_max_seq_len, ctx
    )


def run_ragged_q_both_kv_types[
    num_heads: Int
](
    name: StringLiteral,
    cache_lengths: List[Int],
    seq_lens: List[Int],
    ctx: DeviceContext,
) raises:
    """Run ragged-Q correctness test for both bf16 and fp8 KV types."""
    run_test_paged_variable_ragged_q[.bfloat16, DType.bfloat16, num_heads](
        name + "_bf16", cache_lengths, seq_lens, ctx
    )
    run_test_paged_variable_ragged_q[
        DType.bfloat16, DType.float8_e4m3fn, num_heads
    ](name + "_fp8", cache_lengths, seq_lens, ctx)


def run_all_three_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run correctness test for bf16, fp8 (converter), and native fp8."""
    run_both_kv_types[num_heads](name, cache_lengths, ctx)
    run_test_paged_variable_native_fp8[num_heads](
        name + "_native_fp8", cache_lengths, ctx
    )


def run_uniform_all_three[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run correctness test with uniform cache lengths for all three KV types.
    """
    run_all_three_kv_types[num_heads](name, make_uniform(count, value), ctx)


def run_bench_all_three_kv_types[
    num_heads: Int
](name: StringLiteral, cache_lengths: List[Int], ctx: DeviceContext) raises:
    """Run benchmark for bf16, fp8 (converter), and native fp8."""
    run_bench_both_kv_types[num_heads](name, cache_lengths, ctx)
    run_bench_paged_variable_native_fp8[num_heads](
        name + "_native_fp8", cache_lengths, ctx
    )


def run_bench_uniform_all_three[
    num_heads: Int
](name: StringLiteral, count: Int, value: Int, ctx: DeviceContext) raises:
    """Run benchmark with uniform cache lengths for all three KV types."""
    run_bench_all_three_kv_types[num_heads](
        name, make_uniform(count, value), ctx
    )


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
                # DeepSeek-V2/R1 full: 128 heads, depth=576
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Paged Variable-Length BENCHMARK (B200)")
                print("=" * 72)
                print()

                # Batch size 1: single long context
                run_bench_uniform_all_three[128]("bs1_32k", 1, 32768, ctx)

                # Batch size 1: medium context
                run_bench_uniform_all_three[128]("bs1_4k", 1, 4096, ctx)

                # Batch size 2: variable lengths
                var b2: List[Int] = [4096, 32768]
                run_bench_all_three_kv_types[128]("bs2_4k_32k", b2, ctx)

                # Batch size 4: mixed lengths
                var b4: List[Int] = [1024, 4096, 16384, 32768]
                run_bench_all_three_kv_types[128]("bs4_mixed", b4, ctx)

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
                run_bench_all_three_kv_types[128]("bs8_mixed", b8, ctx)

                # Batch size 8: all long (worst case)
                run_bench_uniform_all_three[128]("bs8_all32k", 8, 32768, ctx)

                print("=" * 72)
                print("BENCHMARK COMPLETE")
                print("=" * 72)

            else:
                # -----------------------------------------------------------
                # Correctness mode: numerical verification against reference
                # -----------------------------------------------------------
                print("=" * 72)
                print("MLA Decode Paged Variable-Length Test (B200)")
                print("=" * 72)
                print()

                # -----------------------------------------------------------
                # Group 1: Small caches with reference check (16 heads)
                # Both bf16 and fp8 KV types are tested to exercise both
                # mla_decode_sm100_kv_bf16 and mla_decode_sm100_kv_fp8
                # kernels.
                # -----------------------------------------------------------
                print(
                    "--- Group 1: Small caches with reference check"
                    " (16 heads) ---"
                )

                var cl1: List[Int] = [30, 50, 80, 100]
                run_both_kv_types[16]("short_uniform_q1", cl1, ctx)

                var cl2: List[Int] = [30, 256, 640, 1024]
                run_both_kv_types[16]("variable_cache_q1", cl2, ctx)

                # -----------------------------------------------------------
                # Group 2: Large caches with reference check (16 heads)
                # -----------------------------------------------------------
                print(
                    "--- Group 2: Large caches with reference check"
                    " (16 heads) ---"
                )

                var cl4: List[Int] = [30, 1024, 8192, 32768]
                run_both_kv_types[16]("extreme_disparity_q1", cl4, ctx)

                var cl6: List[Int] = [
                    30,
                    128,
                    256,
                    512,
                    1024,
                    4096,
                    16384,
                    32768,
                ]
                run_both_kv_types[16]("mixed_8batch_q1", cl6, ctx)

                # -----------------------------------------------------------
                # Group 3: Higher head counts (64, 128) with verification
                # -----------------------------------------------------------
                print(
                    "--- Group 3: Higher head counts with reference check ---"
                )

                var cl7: List[Int] = [30, 512, 4096, 16384]
                run_both_kv_types[64]("64heads_disparity_q1", cl7, ctx)

                var cl8: List[Int] = [30, 1024, 8192, 32768]
                run_both_kv_types[128]("128heads_extreme_q1", cl8, ctx)

                # -----------------------------------------------------------
                # Group 4: Latency sensitive configs (uniform cache lengths)
                # Matches bench_mla_decode.yaml latency sensitive section
                # -----------------------------------------------------------
                print("--- Group 4: Latency sensitive configs (16 heads) ---")

                run_uniform_both[16]("lat_bs1_64k", 1, 65536, ctx)
                run_uniform_both[16]("lat_bs4_32k", 4, 32768, ctx)
                run_uniform_both[16]("lat_bs8_64k", 8, 65536, ctx)

                # -----------------------------------------------------------
                # Group 5: Production-representative configs
                # SGLang/Ant Group: 12-48 per GPU, vLLM TP8: up to 128
                # KV cache 4K-8K is the common production range
                # -----------------------------------------------------------
                print(
                    "--- Group 5: Production-representative configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("prod_bs16_4k", 16, 4096, ctx)
                run_uniform_both[16]("prod_bs32_8k", 32, 8192, ctx)
                run_uniform_both[16]("prod_bs128_1k", 128, 1024, ctx)

                # -----------------------------------------------------------
                # Group 6: Extra-large batch stress test configs
                # -----------------------------------------------------------
                print(
                    "--- Group 6: Extra-large batch stress test configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("stress_bs256_4k", 256, 4096, ctx)

                # -----------------------------------------------------------
                # Group 7: Very small cache_len (<1K)
                # Tests the kernel with very few KV pages -- potential
                # underutilization. Covers prefill-like decode scenarios
                # with tiny cache and high batch.
                # -----------------------------------------------------------
                print(
                    "--- Group 7: Very small cache_len configs (16 heads) ---"
                )

                run_uniform_both[16]("tiny_bs128_64", 128, 64, ctx)
                run_uniform_both[16]("tiny_bs64_512", 64, 512, ctx)

                # -----------------------------------------------------------
                # Group 8: Mid-range configs (fills 8K->32K gap with 16K)
                # -----------------------------------------------------------
                print("--- Group 8: Mid-range configs (16 heads) ---")

                run_uniform_both[16]("mid_bs8_16k", 8, 16384, ctx)

                # -----------------------------------------------------------
                # Group 9: Large cache_len (>64K up to 163K max context)
                # Tests deep into long-context territory for DeepSeek.
                # -----------------------------------------------------------
                print("--- Group 9: Large cache_len configs (16 heads) ---")

                run_uniform_both[16]("large_bs1_163k", 1, 163840, ctx)

                # -----------------------------------------------------------
                # Group 10: Non-power-of-2 cache_len configs
                # Validates that kernel dispatch, page-level partitioning,
                # and split-k reduction handle arbitrary cache lengths.
                # -----------------------------------------------------------
                print(
                    "--- Group 10: Non-power-of-2 cache_len configs"
                    " (16 heads) ---"
                )

                run_uniform_both[16]("npo2_bs16_7777", 16, 7777, ctx)
                run_uniform_both[16]("npo2_bs2_50000", 2, 50000, ctx)

                # -----------------------------------------------------------
                # Group 11: Variable cache_len across batch (log-spaced)
                # Tests the kernel with heterogeneous cache lengths in a
                # single batch, sampled in log space from 128 to 32768.
                # This exercises the variable-length dispatch path,
                # page-level partitioning with different split counts per
                # sequence, and the combine kernel with mixed workloads.
                # -----------------------------------------------------------
                print(
                    "--- Group 11: Variable cache_len across batch"
                    " (log-spaced, 16 heads) ---"
                )

                # 10 sequences with log-spaced cache lengths from 128 to
                # 32768.
                var variable_logspace: List[Int] = [
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    24576,
                    32768,
                ]
                run_both_kv_types[16](
                    "variable_logspace", variable_logspace, ctx
                )

                # -----------------------------------------------------------
                # Group 12: Multi-Q token tests (q_max_seq_len > 1)
                # Tests the ragged layout path where block_idx.y iterates
                # over multiple query tokens per batch entry.  Verifies:
                # - Q/output row offset arithmetic for multi-token decode
                # - o_accum_split padded seq dimension in combine kernel
                # - FP8 blockwise scaling is independent of Q seq_len
                # -----------------------------------------------------------
                print(
                    "--- Group 12: Multi-Q token tests (q_max_seq_len > 1) ---"
                )

                # seq_len=2, small batch, short cache (16 heads)
                var mq1: List[Int] = [30, 50, 80, 100]
                run_multiq_both_kv_types[16]("multiq2_short", mq1, 2, ctx)

                # seq_len=4, mixed cache lengths (16 heads)
                var mq2: List[Int] = [30, 256, 640, 1024]
                run_multiq_both_kv_types[16]("multiq4_mixed", mq2, 4, ctx)

                # seq_len=3, extreme disparity (16 heads)
                var mq3: List[Int] = [30, 1024, 8192, 32768]
                run_multiq_both_kv_types[16]("multiq3_extreme", mq3, 3, ctx)

                # seq_len=2, 128 heads (full DeepSeek config)
                var mq4: List[Int] = [30, 1024, 8192, 32768]
                run_multiq_both_kv_types[128]("multiq2_128heads", mq4, 2, ctx)

                # -----------------------------------------------------------
                # Group 13: Truly ragged Q — each batch has a DIFFERENT
                # number of query tokens.
                # This is the ONLY test that exercises the early-exit path
                # in OffsetPosition where block_idx.y >= per-batch seq_len,
                # and where num_keys = cache_len + per-batch seq_len (NOT
                # + q_max_seq_len).
                # -----------------------------------------------------------
                print(
                    "--- Group 13: Ragged Q — variable per-batch Q seq_len ---"
                )

                # Basic ragged: seq_lens=[1,3,2,4], mixed cache (16 heads)
                var rq_cl1: List[Int] = [256, 1024, 512, 2048]
                var rq_sl1: List[Int] = [1, 3, 2, 4]
                run_ragged_q_both_kv_types[16](
                    "ragged_basic", rq_cl1, rq_sl1, ctx
                )

                # With 128 heads (full DeepSeek config)
                var rq_cl3: List[Int] = [256, 4096, 1024, 8192]
                var rq_sl3: List[Int] = [2, 1, 5, 3]
                run_ragged_q_both_kv_types[128](
                    "ragged_128heads", rq_cl3, rq_sl3, ctx
                )

                # All-one seq_len except last (tests early-exit for most
                # CTAs in the Y dimension)
                var rq_cl4: List[Int] = [
                    30,
                    256,
                    640,
                    1024,
                    4096,
                    8192,
                ]
                var rq_sl4: List[Int] = [1, 1, 1, 1, 1, 6]
                run_ragged_q_both_kv_types[16](
                    "ragged_mostly_1", rq_cl4, rq_sl4, ctx
                )

                # -----------------------------------------------------------
                # Group 14: Native FP8 (Q=FP8 + KV=FP8, output=BF16)
                # Tests the 3-WG native FP8 kernel through the paged path
                # with split-K + PDL. Q is quantized to FP8 before being
                # sent to the kernel. The reference uses BF16 Q and K.
                # -----------------------------------------------------------
                print(
                    "--- Group 14: Native FP8 paged"
                    " (Q=FP8, KV=FP8, output=BF16) ---"
                )

                # Small caches (16 heads)
                var nfp8_1: List[Int] = [30, 50, 80, 100]
                run_test_paged_variable_native_fp8[16](
                    "native_fp8_short_16h", nfp8_1, ctx
                )

                # Variable cache lengths (16 heads)
                var nfp8_2: List[Int] = [30, 256, 640, 1024]
                run_test_paged_variable_native_fp8[16](
                    "native_fp8_variable_16h", nfp8_2, ctx
                )

                # Extreme disparity (16 heads)
                var nfp8_3: List[Int] = [30, 1024, 8192, 32768]
                run_test_paged_variable_native_fp8[16](
                    "native_fp8_extreme_16h", nfp8_3, ctx
                )

                # Full DeepSeek config (128 heads)
                var nfp8_4: List[Int] = [30, 1024, 8192, 32768]
                run_test_paged_variable_native_fp8[128](
                    "native_fp8_extreme_128h", nfp8_4, ctx
                )

                # Production-like: moderate batch, medium cache (128 heads)
                run_test_paged_variable_native_fp8[128](
                    "native_fp8_prod_bs8_4k",
                    make_uniform(8, 4096),
                    ctx,
                )

                # Large batch, short cache (128 heads)
                run_test_paged_variable_native_fp8[128](
                    "native_fp8_bs64_1k",
                    make_uniform(64, 1024),
                    ctx,
                )

                print("=" * 72)
                print("ALL TESTS PASSED")
                print("=" * 72)
        elif has_amd_gpu_accelerator():
            # ---------------------------------------------------------------
            # AMD MI355 / gfx950 path. The AMD MLA decode kernel supports
            # single-token decode (S=1) for any num_heads; the MTP token fold
            # (S = q_seq_len > 1) is FP8-only and requires num_heads <= 16
            # (Kimi-K2.5 TP=4). qkv_fp8's `test_decoding_fold` only covers the
            # contiguous-K dense uniform path — this exercises the fold through
            # the production paged KV cache + ragged input-row-offsets layout.
            #
            # Why a separate block from the NVIDIA SM100 path above (not just
            # reusing its groups): the *helpers* are shared — Group A drives the
            # same `run_test_paged_variable` / `run_test_paged_variable_native_
            # fp8` that the NVIDIA groups use. Only the config lists are disjoint,
            # for backend reasons: the NVIDIA groups cover num_heads up to 128
            # (full DeepSeek) with bf16 + fp8-converter KV, neither of which is in
            # the AMD MTP fold envelope (FP8-only, num_heads <= 16). Running them
            # on AMD would route S=1 (no fold) or hit the host `raise`. The AMD-
            # specific value-add is the ragged/paged multi-Q fold (Groups B/C) —
            # the production layout no NVIDIA group exercises.
            # ---------------------------------------------------------------
            print("=" * 72)
            print("MLA Decode Paged Variable-Length Test (AMD MI355/gfx950)")
            print("=" * 72)
            print()

            # -----------------------------------------------------------
            # Group A: S=1 paged decode (single query token per batch).
            # Confirms the AMD paged decode path works through this harness
            # for both BF16 and native-FP8 (Q=FP8, KV=FP8) decode.
            # -----------------------------------------------------------
            print("--- Group A: S=1 paged decode (16 heads) ---")

            var amd_cl: List[Int] = [30, 256, 640, 1024]
            run_test_paged_variable[.bfloat16, DType.bfloat16, 16](
                "amd_s1_bf16", amd_cl, ctx
            )
            run_test_paged_variable_native_fp8[16](
                "amd_s1_native_fp8", amd_cl, ctx
            )

            # -----------------------------------------------------------
            # Group B: uniform multi-Q FP8 fold (S>1, every sequence has
            # exactly q_max_seq_len query tokens). M = num_heads*S <= 128 →
            # supported. This is the AMD MTP fold through the paged + ragged
            # layout path. Expected to PASS.
            # -----------------------------------------------------------
            print("--- Group B: uniform multi-Q FP8 fold (16 heads) ---")

            var amd_sl_s2: List[Int] = [2, 2, 2, 2]  # M=32, warp-local (2,1)
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_uniform_s2", amd_cl, amd_sl_s2, NullMask(), ctx
            )

            var amd_sl_s4: List[Int] = [4, 4, 4, 4]  # M=64, warp-local (4,1)
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_uniform_s4", amd_cl, amd_sl_s4, NullMask(), ctx
            )

            # -----------------------------------------------------------
            # Group C: VARIABLE ragged-Q FP8 fold — POSITIVE correctness.
            # Per-batch query length differs while q_max_seq_len is only the
            # padding ceiling; the runtime per-sequence length drives the SRD
            # clamp, causal offset, split-K stat writer, and ragged reducer, so
            # short sequences' pad rows are dropped and live rows land at the
            # right slot. The reference (mha_gpu_naive per batch) is fully
            # variable-aware. Covers both np=1 (small caches) and split-K (np>1,
            # large caches — the reducer's ragged remap + pad-row skip).
            #
            # Both masks are run: NullMask validates the ragged remap / reduce /
            # pad-row skip, and CausalMask validates the per-token causal offset
            # `score_row = num_keys - seq_len + token`. Causal is the only mask
            # that *discriminates* the runtime per-sequence seq_len from the
            # comptime q_max_seq_len ceiling (under NullMask the offset never
            # affects output), so the variable configs below are the only place
            # that per-token offset is exercised with a non-uniform seq_len.
            # -----------------------------------------------------------
            print(
                "--- Group C: variable ragged-Q FP8 fold correctness (16"
                " heads) ---"
            )

            # np=1: tiny caches (1 page each) so the heuristic returns
            # num_partitions=1 → the decode kernel writes the ragged output
            # directly, not via the reducer.
            var amd_cl_var_tiny: List[Int] = [30, 64, 96, 120]
            var amd_sl_var: List[Int] = [1, 3, 2, 4]
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_np1_tiny", amd_cl_var_tiny, amd_sl_var, NullMask(), ctx
            )
            # CAUSAL np=1: per-token causal offset on the direct ragged-write
            # path (no reducer). seq_lens=[1,3,2,4] so the runtime per-sequence
            # length != q_max_seq_len=4 for 3 of 4 batches — the discriminating
            # case a comptime-seq_len regression would fail.
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_np1_tiny_causal",
                amd_cl_var_tiny,
                amd_sl_var,
                CausalMask(),
                ctx,
            )

            # np>1 (small): small caches, variable seq_lens (M up to 64).
            var amd_cl_var_small: List[Int] = [256, 1024, 512, 2048]
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_npk_small",
                amd_cl_var_small,
                amd_sl_var,
                NullMask(),
                ctx,
            )
            # CAUSAL np>1: same variable seq_lens through the row-keyed split-K
            # reducer's ragged remap — validates the per-token causal offset is
            # preserved across the reduce (a head-keyed or comptime-seq_len bug
            # would mismatch the per-batch causal reference).
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_npk_small_causal",
                amd_cl_var_small,
                amd_sl_var,
                CausalMask(),
                ctx,
            )

            # np>1: large caches force split-K; variable seq_lens exercise the
            # reducer's ragged output remap + short-sequence pad-row skip.
            var amd_cl_var_big: List[Int] = [256, 65536, 1024, 32768]
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_npk_mixed", amd_cl_var_big, amd_sl_var, NullMask(), ctx
            )

            # np>1, uniform-large cache, variable seq_lens (every batch deep
            # context → every batch split-K; ragged Q row count varies).
            var amd_cl_var_uniform_big: List[Int] = [
                65536,
                65536,
                65536,
                65536,
            ]
            var amd_sl_var_b: List[Int] = [1, 4, 2, 3]
            run_test_paged_variable_ragged_q_native_fp8[16](
                "amd_var_npk_uniformcache",
                amd_cl_var_uniform_big,
                amd_sl_var_b,
                NullMask(),
                ctx,
            )

            # H=8 variable case: DEFERRED. It passes under the JIT `mojo run`
            # path, but the H=8 ragged ladder instantiates all S=1..8 kernels and
            # the S=6 fold (`_48x128x128_True_nqh8`) crashes the LLVM register
            # allocator under the AOT `--config=kernels` build (a pre-existing
            # backend bug — the dense `_False_nqh8` twin compiles fine; only the
            # ragged variant tips regalloc over its VGPR threshold). Re-enable
            # once that's fixed; production fold is num_heads=16, covered above.

            print("=" * 72)
            print("ALL TESTS PASSED")
            print("=" * 72)
        else:
            print("Skipping: requires B200 or AMD GPU")
