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

"""Test MLA_SM100_Decode_Sparse kernel with split KV TMA descriptors.

The sparse kernel splits the KV cache into two separate TMA loads:
  - k_nope_tma: INT64, 64x80, SWIZZLE_NONE  (content/nope part, linear SMEM)
  - k_rope_tma: BF16, 64x64, SWIZZLE_128B   (rope part)

Q stays as a single BF16 TMA (64x576, SWIZZLE_128B).

The KV cache physical layout per token row:
  [512 bytes FP8 nope] [128 bytes BF16 rope]  = 640 bytes total

This test:
  1. Creates a paged KV cache with the sparse layout (head_size=640).
  2. Fills nope (FP8) and rope (BF16) regions with random data.
  3. Creates split TMA descriptors for nope and rope.
  4. Launches MLA_SM100_Decode_Sparse.kernel() directly.
  5. Compares against a naive host-side reference matmul (Q * K^T, softmax, * V).
"""

from std.math import ceildiv, exp
from std.memory import alloc, bitcast
from std.random import randn, seed
from std.sys import argv, has_nvidia_gpu_accelerator, size_of

from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
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
from layout.tma_async import (
    SplitLastDimTMATensorTile,
    TMATensorTile,
    _gather4_box_width,
    create_split_tma,
    create_tma_tile_gather4,
)
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    LayoutTensorMHAOperand,
    MHAOperand,
)
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm90.attention import KVTMATile
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_Decode_Pack,
    QOTMATile,
    tma_tile_qo,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_sparse import (
    MLA_SM100_Decode_Sparse,
)
from nn.attention.gpu.nvidia.sm90.attention import NullPointer
from std.testing import assert_almost_equal
from std.utils.index import Index, IndexList
from std.utils.numerics import min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# MLA dimensions (matching DeepSeek V3 production config).
comptime Q_DEPTH = 576  # Full Q depth: 512 nope + 64 rope
comptime V_DEPTH = 512  # Output depth (nope only)
comptime ROPE_DEPTH = 64  # Rope dimension
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head

# KV cache row width in FP8 bytes: 512 (FP8 nope) + 128 (BF16 rope) = 640.
# We model this as head_size=640 in the KV cache with dtype=float8_e4m3fn
# so that each row has 640 FP8-typed slots. The first 512 are true FP8 data;
# the last 128 are reinterpreted as 64 BF16 elements by the rope TMA.
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH * 2  # 640


def _gcd(a: Int, b: Int) -> Int:
    """Compute greatest common divisor (Euclidean algorithm)."""
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Find a multiplier coprime to n for use in (p * m + 1) % n permutation.

    Tries 3, 5, 7, 11, 13 in order. For n >= 2 at least one will work
    since n cannot be divisible by all of {3, 5, 7, 11, 13}.
    """
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


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# ===-----------------------------------------------------------------------===#
# Host-side reference: BF16 Q (576) x combined K^T (576) -> P (64x64) -> O
# ===-----------------------------------------------------------------------===#


def host_reference[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_bf16_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,  # Q_DEPTH = 576
    v_depth: Int,  # V_DEPTH = 512
    scale: Float32,
    q_max_seq_len: Int = 1,
):
    """Compute reference MLA output on host.

    Q: [batch_size * q_max_seq_len, num_heads, depth(576)]  (ragged layout)
    K: [batch_size, num_keys, depth(576)] in BF16
    V = K[:, :, :v_depth]  (first 512 dims of K)

    Each of the q_max_seq_len query tokens in a batch independently
    attends to the same K/V cache.

    For each batch b, seq s, head h:
      S[k] = sum_d Q[b*seq+s,h,d] * K[b,k,d] * scale
      P[k] = softmax(S)[k]
      O[d] = sum_k P[k] * V[b,k,d]
    """
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                var q_base = (
                    b * q_max_seq_len * num_heads * depth
                    + s * num_heads * depth
                    + h * depth
                )

                # Compute S = Q * K^T * scale
                var max_s = Float64(min_or_neg_inf[.float32]())
                var s_buf = List(length=num_keys, fill=Float64(0))
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    var dot = Float64(0)
                    for d in range(depth):
                        dot += (
                            q_ptr[q_base + d].cast[.float64]()
                            * k_bf16_ptr[k_base + d].cast[.float64]()
                        )
                    s_buf[k] = dot * Float64(scale)
                    if s_buf[k] > max_s:
                        max_s = s_buf[k]

                # Softmax
                var sum_exp = Float64(0)
                for k in range(num_keys):
                    s_buf[k] = exp(s_buf[k] - max_s)
                    sum_exp += s_buf[k]
                for k in range(num_keys):
                    s_buf[k] = s_buf[k] / sum_exp

                # O = P * V (V = first v_depth dims of K)
                var o_base = (
                    b * q_max_seq_len * num_heads * v_depth
                    + s * num_heads * v_depth
                    + h * v_depth
                )
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(num_keys):
                        var k_base = b * num_keys * depth + k * depth
                        acc += (
                            s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()


# ===-----------------------------------------------------------------------===#
# Core test function
# ===-----------------------------------------------------------------------===#


def run_test_sparse[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
) raises:
    """Test the sparse MLA decode kernel.

    Only topk randomly-selected tokens are included in d_indices.
    The kernel iterates over topk entries (not the full cache).
    """
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
    )

    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)
    comptime group = num_heads

    # The KV cache uses kv_type=float8_e4m3fn with head_size=640.
    # Physical layout per token row: 512 FP8 nope | 128 bytes BF16 rope.
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

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

    # Allocate KV cache on host.
    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    # Zero-initialize the entire cache.
    # Generate random BF16 data for nope (512 dims) and rope (64 dims).
    # We store the combined BF16 view for reference computation.
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = List(length=k_bf16_total, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # Fill the KV cache blocks with the sparse layout:
    #   [nope: 512 FP8 bytes] [rope: 64 BF16 = 128 bytes]
    # Token stride in the KV cache = head_size = 640 FP8 slots.
    var tok_stride = kv_params.head_size  # 640 FP8 slots

    # Build lookup table with SHUFFLED page mapping.
    # Instead of identity (page i -> page i), we use a deterministic
    # permutation: physical = (logical * 3 + 1) % num_pages_for_batch.
    # This ensures gather4 TMA loads from non-contiguous physical rows,
    # actually exercising the scatter access pattern.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            # Deterministic permutation using coprime multiplier.
            # (p * mult + 1) % np is a bijection when gcd(mult, np) == 1.
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np
    # Print the page mapping to verify non-identity shuffle.
    if batch_size <= 4:
        for bi in range(batch_size):
            var np = ceildiv(num_keys, PAGE_SIZE)
            print("  batch", bi, "page mapping:", end="")
            for p in range(np):
                print(
                    " ",
                    p,
                    "->",
                    lookup_table_host[bi * max_pages_per_batch + p],
                    end="",
                )
            print()

    # Cache lengths (each batch has cache_len tokens cached).
    var cache_lengths_host = List(length=batch_size, fill=UInt32(cache_len))

    # Fill KV cache with sparse layout data.
    # For each batch b, token t: write FP8 nope then BF16 rope.
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

            # k_bf16 index: [batch, token, dim]
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            for d in range(V_DEPTH):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

            # Write rope (last 64 dims): BF16, stored as raw bytes.
            # Offset in FP8 slots: V_DEPTH (512). Each BF16 = 2 FP8 slots.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]

    # Also create a BF16 version of K for reference (cast nope FP8 back).
    # The reference needs the exact FP8-rounded values for nope.
    var k_ref_host = List(length=k_bf16_total, fill=Scalar[q_type](0))
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH

            # Read back nope as FP8 -> BF16 (captures quantization error).
            for d in range(V_DEPTH):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

            # Read back rope as BF16 (exact, no conversion loss).
            var rope_ptr_bf16 = (blocks_host_ptr + base + V_DEPTH).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                k_ref_host[k_base + V_DEPTH + d] = rope_ptr_bf16[d]

    # -----------------------------------------------------------------------
    # Q tensor: [batch_size * q_max_seq_len, num_heads, Q_DEPTH] (ragged)
    # -----------------------------------------------------------------------
    var q_size = batch_size * q_max_seq_len * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Select topk tokens per batch (sparse subset of all num_keys tokens).
    # Use a deterministic permutation seeded by batch index to pick which
    # tokens are included.  selected_tokens[bi][i] = logical token index.
    # -----------------------------------------------------------------------
    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        # Build a permuted list of all token indices and take first topk.
        # Permutation: (t * mult + 1) % num_keys  (bijection when coprime).
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            # Pick token i via permutation — guaranteed unique.
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    # -----------------------------------------------------------------------
    # Build sparse reference K buffer: [batch_size, topk, Q_DEPTH]
    # Contains the FP8-rounded K values for only the selected tokens.
    # -----------------------------------------------------------------------
    var k_sparse_ref_size = batch_size * topk * Q_DEPTH
    var k_sparse_ref = List(length=k_sparse_ref_size, fill=Scalar[q_type](0))
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            var dst_base = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]

    # -----------------------------------------------------------------------
    # Reference output on host (using only the selected topk tokens)
    # -----------------------------------------------------------------------
    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
    host_reference[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        topk,
        Q_DEPTH,
        V_DEPTH,
        scale,
        q_max_seq_len,
    )

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollectionT on device
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
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for the selected topk tokens only.
    # d_indices[batch * topk + i] = physical_block * PAGE_SIZE + offset.
    # The kernel internally divmods to recover (block, offset) and computes
    # the TMA row as block * kv_stride + offset.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * topk
    var h_indices = List(length=total_indices, fill=Int32(0))
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            # Encode as physical_block * page_size + offset (FlashMLA convention).
            # The kernel will divmod to get (block_id, tok_in_page) and compute
            # block_id * kv_stride + tok_in_page for the actual TMA row.
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch
    # -----------------------------------------------------------------------
    # Q: [batch_size * q_max_seq_len, num_heads, Q_DEPTH] (rank 3, ragged)
    var total_q_tokens = batch_size * q_max_seq_len
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[Q_DEPTH])),
    )

    # Output: [batch_size * q_max_seq_len, num_heads, V_DEPTH]
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    # Each batch has q_max_seq_len query tokens.
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(batch_size + 1),
    )

    # Scalar args: [batch_size, q_max_seq_len, num_partitions]
    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    # Compute and print num_partitions to verify split-K usage.
    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    # Indices stride = topk (number of entries per batch).
    var indices_stride = topk

    print(
        "  Launching MLA decode kernel (sparse, through dispatch)...",
        " topk=",
        topk,
        " num_keys=",
        num_keys,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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

    # -----------------------------------------------------------------------
    # Verify output
    # -----------------------------------------------------------------------
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # Compare against reference.
    # FP8 quantization + BF16 accumulation => moderate tolerance.
    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
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
                    var err = abs(actual_val - ref_val)
                    if err > max_err:
                        max_err = err
                    total_checked += 1
                    if err > 1e-1:
                        print(b, s, h, d, actual_val, ref_val, err)
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

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = row_offsets_host^
    _ = h_indices^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = selected_tokens^
    _ = k_sparse_ref^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


# ===-----------------------------------------------------------------------===#
# Host-side reference with blockwise FP8 scaling
# ===-----------------------------------------------------------------------===#


def host_reference_blockscale[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_bf16_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,  # Q_DEPTH = 576
    v_depth: Int,  # V_DEPTH = 512
    scale: Float32,
    q_max_seq_len: Int = 1,
):
    """Compute reference MLA output on host with variable-length batches.

    Q: [batch_size * q_max_seq_len, num_heads, depth(576)]  (ragged layout)
    K: stored per-batch contiguously in k_bf16_ptr with variable num_keys
    V = K[:, :, :v_depth]  (first 512 dims of K)

    Each of the q_max_seq_len query tokens in a batch independently
    attends to the same K/V cache.

    For each batch b, seq s, head h:
      S[k] = sum_d Q[b*seq+s,h,d] * K[b,k,d] * scale
      P[k] = softmax(S)[k]
      O[d] = sum_k P[k] * V[b,k,d]
    """
    # Compute the offset into the K buffer for each batch entry.
    var k_offsets = List(length=batch_size, fill=Int(0))
    var running = 0
    for b in range(batch_size):
        k_offsets[b] = running
        running += num_keys_per_batch[b] * depth

    for b in range(batch_size):
        var num_keys = num_keys_per_batch[b]
        var k_base_offset = k_offsets[b]
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                var q_base = (
                    b * q_max_seq_len * num_heads * depth
                    + s * num_heads * depth
                    + h * depth
                )

                # Compute S = Q * K^T * scale
                var max_s = Float64(min_or_neg_inf[.float32]())
                var s_buf = List(length=num_keys, fill=Float64(0))
                for k in range(num_keys):
                    var k_base = k_base_offset + k * depth
                    var dot = Float64(0)
                    for d in range(depth):
                        dot += (
                            q_ptr[q_base + d].cast[.float64]()
                            * k_bf16_ptr[k_base + d].cast[.float64]()
                        )
                    s_buf[k] = dot * Float64(scale)
                    if s_buf[k] > max_s:
                        max_s = s_buf[k]

                # Softmax
                var sum_exp = Float64(0)
                for k in range(num_keys):
                    s_buf[k] = exp(s_buf[k] - max_s)
                    sum_exp += s_buf[k]
                for k in range(num_keys):
                    s_buf[k] = s_buf[k] / sum_exp

                # O = P * V (V = first v_depth dims of K)
                var o_base = (
                    b * q_max_seq_len * num_heads * v_depth
                    + s * num_heads * v_depth
                    + h * v_depth
                )
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(num_keys):
                        var k_base = k_base_offset + k * depth
                        acc += (
                            s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()
                _ = s_buf^
    _ = k_offsets^


def _palette_scale(index: Int) -> Float32:
    """Pick a power-of-2 scale from the palette by index (wrapping)."""
    if index % 7 == 0:
        return 0.25
    if index % 7 == 1:
        return 0.5
    if index % 7 == 2:
        return 1.0
    if index % 7 == 3:
        return 2.0
    if index % 7 == 4:
        return 4.0
    if index % 7 == 5:
        return 0.125
    return 8.0


# ===-----------------------------------------------------------------------===#
# Test: sparse kernel with blockwise FP8 scaling + variable paged KV cache
# ===-----------------------------------------------------------------------===#


def run_test_sparse_blockscale[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    scale_block_size: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    ctx: DeviceContext,
    topk: Int,
) raises:
    """Test sparse kernel with blockwise FP8 scaling and variable paged KV cache.

    Each batch entry can have a different cache length. Scales are non-uniform
    power-of-2 values per block per token.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)
    comptime group = num_heads
    # scales_per_token = ceildiv(Q_DEPTH, scale_block_size)
    comptime scales_per_token = ceildiv(Q_DEPTH, scale_block_size)

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
        " scale_block_size:",
        scale_block_size,
        " scales_per_token:",
        scales_per_token,
    )
    for i in range(batch_size):
        print("  batch", i, ": cache_len=", cache_lengths[i])

    # Compute max cache_len, total pages, and per-batch num_keys.
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

    # The KV cache uses kv_type=float8_e4m3fn with head_size=640.
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
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
    var tok_stride = kv_params.head_size  # 640 FP8 slots per token

    # -----------------------------------------------------------------------
    # Allocate KV cache blocks and zero-initialize
    # -----------------------------------------------------------------------
    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    # -----------------------------------------------------------------------
    # Build lookup table with SHUFFLED page mapping.
    # Deterministic permutation: physical = (logical * 3 + 1) % num_pages.
    # This ensures gather4 TMA loads from non-contiguous physical rows.
    # -----------------------------------------------------------------------
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            # Deterministic permutation using coprime multiplier.
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi
    # Print the page mapping to verify non-identity shuffle.
    if batch_size <= 4:
        for bi in range(batch_size):
            var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
            print("  batch", bi, "page mapping:", end="")
            for p in range(np_bi):
                print(
                    " ",
                    p,
                    "->",
                    lookup_table_host[bi * max_pages_per_batch + p],
                    end="",
                )
            print()

    # Cache lengths
    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # -----------------------------------------------------------------------
    # Allocate scales buffer (flat: total_pages * page_size * scales_per_token)
    # Layout matches the 6D scales tensor:
    #   [total_pages, kv_dim2, NUM_LAYERS, PAGE_SIZE, KV_NUM_HEADS, scales_per_token]
    # Flattened, accessed as: scales_ptr[gmem_row * scales_per_token + s]
    #   where gmem_row = physical_page * PAGE_SIZE + tok_in_page
    # -----------------------------------------------------------------------
    var scales_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * scales_per_token
    )
    # Initialize all scales to 1.0 (neutral)
    var scales_host = List(length=scales_elems, fill=Float32(1.0))

    # Page stride for scales: kv_dim2 * NUM_LAYERS * PAGE_SIZE * KV_NUM_HEADS * scales_per_token
    var scale_page_stride = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * scales_per_token
    )
    var scale_tok_stride = kv_params.num_heads * scales_per_token

    # Page stride for blocks
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Generate random K data (BF16) and fill KV cache + scales
    # -----------------------------------------------------------------------
    # For reference: store all K data per batch contiguously.
    # Total K elements across all batches:
    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH

    var k_bf16_host = List(length=total_k_elems, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # Fill KV cache blocks and scales with non-uniform values.
    # Also build the reference K buffer with block-scale-corrected values.
    var k_ref_host = List(length=total_k_elems, fill=Scalar[q_type](0))

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
            var scale_base = (
                physical_page * scale_page_stride
                + tok_in_page * scale_tok_stride
            )

            # Assign non-uniform per-block, per-token scales.
            # Use palette with token*7+blk to get variation.
            var tok_global = k_offset // Q_DEPTH + t
            for s in range(scales_per_token):
                scales_host[scale_base + s] = _palette_scale(tok_global * 7 + s)

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            # Then dequantize with block scale for reference.
            for d in range(V_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                # Dequantize: fp8_val * block_scale for reference
                var block_idx = d // scale_block_size
                var block_scale = scales_host[scale_base + block_idx]
                k_ref_host[k_base + d] = (
                    fp8_val.cast[.float32]() * block_scale
                ).cast[q_type]()

            # Write rope (last 64 dims): BF16, stored as raw bytes.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]
                # Rope is BF16, no scaling needed for reference.
                k_ref_host[k_base + V_DEPTH + d] = k_bf16_host[
                    k_base + V_DEPTH + d
                ]

        k_offset += nk * Q_DEPTH

    # Zero out tail slots in each page (tokens beyond num_keys for each batch).
    # Must use the shuffled lookup table to find the correct physical page.
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var num_pages_bi = ceildiv(nk, PAGE_SIZE)
        for pg in range(num_pages_bi):
            var valid_toks = nk - pg * PAGE_SIZE
            if valid_toks > PAGE_SIZE:
                valid_toks = PAGE_SIZE
            # Use the lookup table to get the physical page.
            var phys_page = Int(
                lookup_table_host[bi * max_pages_per_batch + pg]
            )
            # Zero out [valid_toks, PAGE_SIZE) in this physical page.
            var base = phys_page * page_stride_elems + valid_toks * tok_stride
            var zero_count = (PAGE_SIZE - valid_toks) * tok_stride
            for z in range(zero_count):
                blocks_host[base + z] = Scalar[kv_type](0)

    # -----------------------------------------------------------------------
    # Q tensor: [batch_size, num_heads, Q_DEPTH]
    # -----------------------------------------------------------------------
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Select topk tokens per batch and build sparse reference K buffer.
    # -----------------------------------------------------------------------
    var selected_tokens_bs = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk):
            selected_tokens_bs[bi * topk + i] = (i * mult + 1) % nk

    # Build sparse reference K: [batch_size * topk * Q_DEPTH] (flat per-batch).
    var k_sparse_ref_size = batch_size * topk * Q_DEPTH
    var k_sparse_ref = List(length=k_sparse_ref_size, fill=Scalar[q_type](0))
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for i in range(topk):
            var t = selected_tokens_bs[bi * topk + i]
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        k_offset_src += nk * Q_DEPTH

    # Sparse num_keys_list: each batch has exactly topk keys.
    var sparse_num_keys_list = List[Int]()
    for _bi in range(batch_size):
        sparse_num_keys_list.append(topk)

    # -----------------------------------------------------------------------
    # Reference output on host (using only the selected topk tokens)
    # -----------------------------------------------------------------------
    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
    host_reference_blockscale[q_type](
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

    # -----------------------------------------------------------------------
    # Copy to device
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

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection with scales
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    var scales_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        scales_per_token,
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
        PAGE_SIZE,
        scale_dtype_=DType.float32,
        quantization_granularity_=scale_block_size,
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
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for selected topk tokens only.
    # d_indices[batch * topk + i] = physical_block * PAGE_SIZE + offset.
    # The kernel internally divmods to recover (block, offset) and computes
    # the TMA row as block * kv_stride + offset.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * topk
    var h_indices_bs = List(length=total_indices, fill=Int32(0))
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens_bs[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            # Encode as physical_block * page_size + offset (FlashMLA convention).
            h_indices_bs[bi * topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )

    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices_bs)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch
    # -----------------------------------------------------------------------
    # Q: [batch_size, num_heads, Q_DEPTH] (rank 3, ragged=True)
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )

    # Output: [batch_size, num_heads, V_DEPTH]
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged: [0, 1, 2, ..., batch_size]
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    # Compute and print num_partitions to verify split-K usage.
    comptime sm_count_bs = ctx.default_device_info.sm_count
    var dispatch_scalars_bs = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count_bs // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count_bs)
    var num_partitions_bs = dispatch_scalars_bs[2]
    print(
        "  num_partitions=",
        num_partitions_bs,
        " (split-K",
        "ACTIVE" if num_partitions_bs > 1 else "OFF",
        ")",
    )

    # Indices stride = topk (number of entries per batch).
    var indices_stride_bs = topk

    print(
        (
            "  Launching MLA decode kernel (sparse blockscale, through"
            " dispatch)..."
        ),
        " topk=",
        topk,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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
        indices_stride=indices_stride_bs,
    )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output
    # -----------------------------------------------------------------------
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # FP8 quantization + blockwise scaling + BF16 accumulation.
    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
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
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = mla_args
    _ = blocks_device
    _ = scales_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = h_indices_bs^
    _ = scales_host^
    _ = row_offsets_host^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = selected_tokens_bs^
    _ = k_sparse_ref^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


# ===-----------------------------------------------------------------------===#
# Test: sparse kernel with per-batch VARIABLE topk
# ===-----------------------------------------------------------------------===#


def run_test_sparse_variable_topk[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Test sparse kernel with variable topk per batch.

    Each batch entry has a different cache length AND a different topk count.
    topk_per_batch[b] = number of valid sparse entries for batch b.
    indices_stride = max(topk_per_batch) = allocation stride.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)
    comptime group = num_heads

    # Compute max topk (allocation stride) and max cache_len.
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
    print("  indices_stride (max topk)=", max_topk)

    # Compute max cache_len, total pages, and per-batch num_keys.
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

    # The KV cache uses kv_type=float8_e4m3fn with head_size=640.
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
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
    var tok_stride = kv_params.head_size  # 640 FP8 slots per token

    # -----------------------------------------------------------------------
    # Allocate KV cache blocks and zero-initialize
    # -----------------------------------------------------------------------
    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    # -----------------------------------------------------------------------
    # Build lookup table with SHUFFLED page mapping.
    # -----------------------------------------------------------------------
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

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
    if batch_size <= 4:
        for bi in range(batch_size):
            var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
            print("  batch", bi, "page mapping:", end="")
            for p in range(np_bi):
                print(
                    " ",
                    p,
                    "->",
                    lookup_table_host[bi * max_pages_per_batch + p],
                    end="",
                )
            print()

    # Cache lengths
    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Page stride for blocks
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Generate random K data (BF16) and fill KV cache
    # -----------------------------------------------------------------------
    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH

    var k_bf16_host = List(length=total_k_elems, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # Build reference K with FP8-rounded nope values.
    var k_ref_host = List(length=total_k_elems, fill=Scalar[q_type](0))

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

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            for d in range(V_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()

            # Write rope (last 64 dims): BF16.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]
                k_ref_host[k_base + V_DEPTH + d] = k_bf16_host[
                    k_base + V_DEPTH + d
                ]

        k_offset += nk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Q tensor: [batch_size, num_heads, Q_DEPTH]
    # -----------------------------------------------------------------------
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Select variable topk tokens per batch and build sparse reference.
    # d_indices layout: [batch_size * max_topk], padded with zeros beyond
    # each batch's actual topk.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * max_topk
    var h_indices = List(length=total_indices, fill=Int32(0))
    # Zero-initialize (padding).
    # Per-batch selected tokens (variable count).
    # We also need sparse reference K for the reference computation.
    # Total sparse ref elements: sum of topk_per_batch[bi] * Q_DEPTH.
    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += topk_per_batch[bi] * Q_DEPTH
    var k_sparse_ref = List(
        length=total_sparse_ref_elems, fill=Scalar[q_type](0)
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            # Build d_indices entry.
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            # Build sparse reference K.
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Reference output on host (per-batch variable topk)
    # -----------------------------------------------------------------------
    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(topk_per_batch[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
    host_reference_blockscale[q_type](
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

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
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

    # Build topk_lengths on host and copy to device.
    var topk_lengths_host = List(length=batch_size, fill=Int32(0))
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
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
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged: [0, 1, 2, ..., batch_size]
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    # Compute and print num_partitions to verify split-K usage.
    comptime sm_count_vt = ctx.default_device_info.sm_count
    var dispatch_scalars_vt = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count_vt // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count_vt)
    var num_partitions_vt = dispatch_scalars_vt[2]
    print(
        "  num_partitions=",
        num_partitions_vt,
        " (split-K",
        "ACTIVE" if num_partitions_vt > 1 else "OFF",
        ")",
    )

    # indices_stride = max topk (allocation stride).
    var indices_stride = max_topk

    print(
        "  Launching MLA decode kernel (sparse, variable topk)...",
        " max_topk=",
        max_topk,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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

    # -----------------------------------------------------------------------
    # Verify output
    # -----------------------------------------------------------------------
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # FP8 quantization + BF16 accumulation => moderate tolerance.
    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
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
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device
    _ = topk_lengths_host^
    _ = h_indices^
    _ = row_offsets_host^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = k_sparse_ref^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


# ===-----------------------------------------------------------------------===#
# Host-side reference with attention sink correction
# ===-----------------------------------------------------------------------===#


def host_reference_with_attn_sink[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_bf16_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    attn_sink_host: Pointer[Float32, _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA with attn_sink correction.

    Same as host_reference but the softmax denominator is adjusted:
      sum_exp += exp(attn_sink[h] - max_s)
    This accounts for the non-selected tokens whose aggregate
    contribution is captured by attn_sink[h] (natural log domain).
    """
    for b in range(batch_size):
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            # Compute S = Q * K^T * scale
            var max_s = Float64(min_or_neg_inf[.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = b * num_keys * depth + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[.float64]()
                        * k_bf16_ptr[k_base + d].cast[.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            # Include attn_sink in max computation.
            var attn_sink_val = Float64(attn_sink_host[h])
            if attn_sink_val > max_s:
                max_s = attn_sink_val

            # Softmax with attn_sink correction in the denominator.
            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            # Add the attn_sink contribution to the denominator.
            sum_exp += exp(attn_sink_val - max_s)

            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            # O = P * V (V = first v_depth dims of K)
            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^


# ===-----------------------------------------------------------------------===#
# Attn sink sparse test
# ===-----------------------------------------------------------------------===#


def run_test_sparse_attn_sink[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
) raises:
    """Test sparse MLA decode with attn_sink correction.

    Tests both split-K and no-split paths. The attn_sink values are
    per-head float32 values in natural log domain representing the
    aggregate contribution of non-selected tokens.
    """
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
    comptime group = num_heads

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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = List(length=k_bf16_total, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    var tok_stride = kv_params.head_size

    # Build shuffled page mapping.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))
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

    # Fill KV cache with sparse layout data.
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

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            for d in range(V_DEPTH):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

            # Write rope (last 64 dims): BF16, stored as raw bytes.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]

    # Re-read k_bf16 from the blocks to match FP8 quantization.
    var k_ref_host = List(length=k_bf16_total, fill=Scalar[q_type](0))
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH

            # Read back nope as FP8 -> BF16 (captures quantization error).
            for d in range(V_DEPTH):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

            # Read back rope as BF16 (exact, no conversion loss).
            var rope_ptr_bf16 = (blocks_host_ptr + base + V_DEPTH).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                k_ref_host[k_base + V_DEPTH + d] = rope_ptr_bf16[d]

    # Select topk tokens.
    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    seed(42)
    for bi in range(batch_size):
        var all_tokens = List(length=num_keys, fill=Int(0))
        for t in range(num_keys):
            all_tokens[t] = t
        # Simple Fisher-Yates partial shuffle.
        for i in range(topk):
            var j_offset = Int(UInt64(i + 1).cast[.uint32]())
            var j = i + (j_offset % (num_keys - i))
            var tmp = all_tokens[i]
            all_tokens[i] = all_tokens[j]
            all_tokens[j] = tmp
            selected_tokens[bi * topk + i] = all_tokens[i]
        _ = all_tokens^

    # Build sparse reference K from selected tokens only.
    var k_sparse_total = batch_size * topk * Q_DEPTH
    var k_sparse_ref = List(length=k_sparse_total, fill=Scalar[q_type](0))
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            var dst = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst + d] = k_ref_host[src + d]

    # Create per-head attn_sink values (natural log domain).
    # Use moderate values that will noticeably affect the softmax.
    var attn_sink_host = List(length=num_heads, fill=Float32(0))
    for h in range(num_heads):
        # Use values between -2.0 and +2.0, varying by head.
        attn_sink_host[h] = Float32(
            -1.0 + 3.0 * Float32(h) / Float32(num_heads)
        )

    # Q
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.3)

    # Reference output with attn_sink correction.
    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
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

    # Upload data to device.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_host = List(length=batch_size, fill=UInt32(cache_len))
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    # Upload attn_sink to device.
    var attn_sink_device = ctx.enqueue_create_buffer[.float32](num_heads)
    ctx.enqueue_copy(attn_sink_device, attn_sink_host)

    ctx.synchronize()

    # Build PagedKVCacheCollectionT on device.
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
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # Build gather4 indices.
    var total_indices = batch_size * topk
    var h_indices = List(length=total_indices, fill=Int32(0))
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

    # Build TileTensors and call flare_mla_decoding.
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
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
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = topk

    print(
        "  Launching MLA decode kernel (sparse + attn_sink)...",
        " topk=",
        topk,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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
        attn_sink_ptr=rebind[MutPointer[Float32, origin=MutAnyOrigin]](
            attn_sink_device.unsafe_ptr()
        ),
    )

    ctx.synchronize()

    # Verify output.
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var max_err = Float64(0)
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
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=5e-2, rtol=1e-2)

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # Cleanup.
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = attn_sink_device
    _ = attn_sink_host^
    _ = row_offsets_host^
    _ = h_indices^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = selected_tokens^
    _ = k_sparse_ref^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


# ===-----------------------------------------------------------------------===#
# Test: sparse kernel with extra KV (always-attend tokens from separate cache)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_extra_kv[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Test sparse MLA decode with extra KV (always-attend) tokens.

    Creates TWO separate KV caches:
      1. Original cache: topk tokens selected from it (sparse).
      2. Extra cache: extra_topk tokens selected from it (always-attend).

    The kernel processes original topk tokens first, then extra_topk tokens,
    in a unified attention loop. The reference concatenates all selected tokens
    from both caches and computes attention over the combined set.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    # Compute max topk for original and extra.
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
    print("  indices_stride (max topk)=", max_topk)
    print("  extra_indices_stride (max extra topk)=", max_extra_topk)

    # -----------------------------------------------------------------------
    # Shared KV cache parameters
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1
    var tok_stride = kv_params.head_size  # 640 FP8 slots per token
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Build ORIGINAL KV cache
    # -----------------------------------------------------------------------
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

    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    # Build shuffled page mapping for original cache.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))
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

    # Cache lengths.
    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Generate random K data and fill original cache.
    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH
    var k_bf16_host = List(length=total_k_elems, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # Build reference K with FP8-rounded nope values.
    var k_ref_host = List(length=total_k_elems, fill=Scalar[q_type](0))

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

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            for d in range(V_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()

            # Write rope (last 64 dims): BF16.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]
                k_ref_host[k_base + V_DEPTH + d] = k_bf16_host[
                    k_base + V_DEPTH + d
                ]
        k_offset += nk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Build EXTRA KV cache
    # -----------------------------------------------------------------------
    var max_extra_cache_len = 0
    var extra_total_pages = 0
    var extra_num_keys_list = List[Int]()
    for i in range(batch_size):
        var ecl = extra_cache_lengths[i]
        if ecl > max_extra_cache_len:
            max_extra_cache_len = ecl
        var enk = ecl + q_max_seq_len
        extra_num_keys_list.append(enk)
        extra_total_pages += ceildiv(enk, PAGE_SIZE)
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

    var extra_blocks_host = List(
        length=extra_block_elems, fill=Scalar[kv_type](0)
    )
    var extra_blocks_host_ptr: Pointer[
        extra_blocks_host.T, origin_of(extra_blocks_host)
    ] = extra_blocks_host.unsafe_ptr()
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[kv_type](0)

    # Build shuffled page mapping for extra cache.
    var extra_lut_size = batch_size * max_extra_pages_per_batch
    var extra_lookup_table_host = List(length=extra_lut_size, fill=UInt32(0))
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

    # Extra cache lengths.
    var extra_cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        extra_cache_lengths_host[i] = UInt32(extra_cache_lengths[i])

    # Generate random K data for extra cache and fill.
    var extra_total_k_elems = 0
    for bi in range(batch_size):
        extra_total_k_elems += extra_num_keys_list[bi] * Q_DEPTH
    var extra_k_bf16_host = List(
        length=extra_total_k_elems, fill=Scalar[q_type](0)
    )
    randn(
        extra_k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    var extra_k_ref_host = List(
        length=extra_total_k_elems, fill=Scalar[q_type](0)
    )

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

            for d in range(V_DEPTH):
                var fp8_val = extra_k_bf16_host[k_base + d].cast[kv_type]()
                extra_blocks_host[base + d] = fp8_val
                extra_k_ref_host[k_base + d] = fp8_val.cast[q_type]()

            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (extra_blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = extra_k_bf16_host[k_base + V_DEPTH + d]
                extra_k_ref_host[k_base + V_DEPTH + d] = extra_k_bf16_host[
                    k_base + V_DEPTH + d
                ]
        ek_offset += enk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Q tensor: [batch_size, num_heads, Q_DEPTH]
    # -----------------------------------------------------------------------
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Select tokens and build sparse indices for both caches.
    # d_indices layout: [batch_size * max_topk], padded with zeros.
    # extra_d_indices layout: [batch_size * max_extra_topk], padded with zeros.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * max_topk
    var h_indices = List(length=total_indices, fill=Int32(0))
    var extra_total_indices = batch_size * max_extra_topk
    var extra_h_indices = List(length=extra_total_indices, fill=Int32(0))
    # Build combined reference K: [batch, topk+extra_topk, Q_DEPTH].
    var total_combined_ref_elems = 0
    for bi in range(batch_size):
        total_combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * Q_DEPTH
    var k_combined_ref = List(
        length=total_combined_ref_elems, fill=Scalar[q_type](0)
    )

    var combined_ref_offset = 0
    var k_offset_src = 0
    var ek_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)

        # Select topk from original cache.
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
            # Copy to combined reference.
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = k_ref_host[src_base + d]

        combined_ref_offset += topk_bi * Q_DEPTH

        # Select extra_topk from extra cache.
        var enk = extra_num_keys_list[bi]
        var extra_topk_bi = extra_topk_per_batch[bi]
        var extra_mult = _coprime_multiplier(enk)
        for i in range(extra_topk_bi):
            var t = (i * extra_mult + 1) % enk
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
            # Copy to combined reference.
            var src_base = ek_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = extra_k_ref_host[src_base + d]

        combined_ref_offset += extra_topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH
        ek_offset_src += enk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Reference output: attend to combined (topk + extra_topk) tokens.
    # -----------------------------------------------------------------------
    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
    host_reference_blockscale[q_type](
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

    # -----------------------------------------------------------------------
    # Copy everything to device
    # -----------------------------------------------------------------------
    # Original cache.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    # Extra cache.
    var extra_blocks_device = ctx.enqueue_create_buffer[kv_type](
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

    # Q and output.
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    # Indices.
    var d_indices_device = ctx.enqueue_create_buffer[.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var extra_d_indices_device = ctx.enqueue_create_buffer[.int32](
        extra_total_indices
    )
    ctx.enqueue_copy(extra_d_indices_device, extra_h_indices)

    # topk_lengths and extra_topk_lengths.
    var topk_lengths_host = List(length=batch_size, fill=Int32(0))
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    var extra_topk_lengths_host = List(length=batch_size, fill=Int32(0))
    for bi in range(batch_size):
        extra_topk_lengths_host[bi] = Int32(extra_topk_per_batch[bi])
    var extra_topk_lengths_device = ctx.enqueue_create_buffer[.int32](
        batch_size
    )
    ctx.enqueue_copy(extra_topk_lengths_device, extra_topk_lengths_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection for ORIGINAL cache
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

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection for EXTRA cache
    # -----------------------------------------------------------------------
    var extra_blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
        kv_type, kv_params, PAGE_SIZE
    ](
        LayoutTensor[kv_type, Layout.row_major[6]()](
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

    # -----------------------------------------------------------------------
    # Build TileTensors and launch through flare_mla_decoding
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count_ek = ctx.default_device_info.sm_count
    var dispatch_scalars_ek = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count_ek // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count_ek)
    var num_partitions_ek = dispatch_scalars_ek[2]
    print(
        "  num_partitions=",
        num_partitions_ek,
        " (split-K",
        "ACTIVE" if num_partitions_ek > 1 else "OFF",
        ")",
    )

    print(
        "  Launching MLA decode kernel (sparse + extra KV)...",
        " max_topk=",
        max_topk,
        " max_extra_topk=",
        max_extra_topk,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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

    # -----------------------------------------------------------------------
    # Verify output
    # -----------------------------------------------------------------------
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
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
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
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
    _ = row_offsets_host^
    _ = extra_topk_lengths_host^
    _ = topk_lengths_host^
    _ = extra_h_indices^
    _ = h_indices^
    _ = extra_cache_lengths_host^
    _ = extra_lookup_table_host^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = k_combined_ref^
    _ = extra_k_ref_host^
    _ = extra_k_bf16_host^
    _ = extra_blocks_host^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


# ===-----------------------------------------------------------------------===#
# Test: topk clamping for first execution (cache_length=0) and small caches
# ===-----------------------------------------------------------------------===#


def run_test_sparse_topk_clamping[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Test topk clamping when topk > actual available tokens.

    The kernel should clamp topk_per_batch[b] to
    min(topk_per_batch[b], actual_tokens[b]) where
    actual_tokens = cache_length + seq_len (with _is_cache_length_accurate=False).

    Key scenarios:
      - cache_length=0 (first decode): actual_tokens=1, topk=64 -> clamped to 1
      - cache_length=5: actual_tokens=6, topk=64 -> clamped to 6
      - cache_length=256: actual_tokens=257, topk=64 -> no clamping needed

    The reference computation uses the clamped topk values.
    Only the first clamped-topk indices point to valid tokens; the rest are
    garbage (but the kernel should never access them due to clamping).
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)
    comptime group = num_heads

    # Compute max topk (allocation stride) and per-batch effective topk.
    var max_topk = 0
    var effective_topk_list = List[Int]()
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        # actual_tokens = cache_length + seq_len (is_cache_length_accurate=False)
        var actual_tokens = cache_lengths[bi] + q_max_seq_len
        # Effective topk = min(topk_per_batch[b], actual_tokens)
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
    print("  indices_stride (max topk)=", max_topk)

    # Compute max cache_len, total pages, and per-batch num_keys.
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

    # The KV cache uses kv_type=float8_e4m3fn with head_size=640.
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
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
    var tok_stride = kv_params.head_size  # 640 FP8 slots per token

    # -----------------------------------------------------------------------
    # Allocate KV cache blocks and zero-initialize
    # -----------------------------------------------------------------------
    var blocks_host = List(length=block_elems, fill=Scalar[kv_type](0))
    var blocks_host_ptr: Pointer[
        blocks_host.T, origin_of(blocks_host)
    ] = blocks_host.unsafe_ptr()
    # -----------------------------------------------------------------------
    # Build lookup table with SHUFFLED page mapping.
    # -----------------------------------------------------------------------
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = List(length=lut_size, fill=UInt32(0))

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
    if batch_size <= 4:
        for bi in range(batch_size):
            var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
            print("  batch", bi, "page mapping:", end="")
            for p in range(np_bi):
                print(
                    " ",
                    p,
                    "->",
                    lookup_table_host[bi * max_pages_per_batch + p],
                    end="",
                )
            print()

    # Cache lengths
    var cache_lengths_host = List(length=batch_size, fill=UInt32(0))
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    # Page stride for blocks
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Generate random K data (BF16) and fill KV cache
    # -----------------------------------------------------------------------
    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH

    var k_bf16_host = List(length=total_k_elems, fill=Scalar[q_type](0))
    randn(
        k_bf16_host,
        mean=0.0,
        standard_deviation=0.5,
    )

    # Build reference K with FP8-rounded nope values.
    var k_ref_host = List(length=total_k_elems, fill=Scalar[q_type](0))

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

            # Write nope (first 512 dims): BF16 -> FP8 cast.
            for d in range(V_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()

            # Write rope (last 64 dims): BF16.
            var rope_base_fp8 = base + V_DEPTH
            var rope_ptr_bf16 = (blocks_host_ptr + rope_base_fp8).bitcast[
                Scalar[q_type]
            ]()
            for d in range(ROPE_DEPTH):
                rope_ptr_bf16[d] = k_bf16_host[k_base + V_DEPTH + d]
                k_ref_host[k_base + V_DEPTH + d] = k_bf16_host[
                    k_base + V_DEPTH + d
                ]

        k_offset += nk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Q tensor: [batch_size, num_heads, Q_DEPTH]
    # -----------------------------------------------------------------------
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = List(length=q_size, fill=Scalar[q_type](0))
    randn(q_host, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Select tokens per batch for d_indices.
    # We place effective_topk valid entries, then fill the rest with -1
    # (garbage) to verify the kernel never reads beyond the clamped topk.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * max_topk
    # Fill with -1 (invalid sentinel) to ensure kernel does NOT read past
    # the clamped topk. If it does, the gather will hit an invalid row.
    var h_indices = List(length=total_indices, fill=Int32(-1))

    # Build sparse reference K for the EFFECTIVE (clamped) topk.
    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += effective_topk_list[bi] * Q_DEPTH
    var k_sparse_ref = List(
        length=total_sparse_ref_elems, fill=Scalar[q_type](0)
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
                t = 0  # Only one token available
            else:
                t = (i * mult + 1) % nk
            # Build d_indices entry.
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            # Build sparse reference K.
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += eff_topk * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    # -----------------------------------------------------------------------
    # Reference output on host (per-batch effective/clamped topk)
    # -----------------------------------------------------------------------
    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(effective_topk_list[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = List(length=out_size, fill=Scalar[q_type](0))
    host_reference_blockscale[q_type](
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

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
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

    # Build topk_lengths on host and copy to device.
    # These are the UNCLAMPED topk values. The kernel must clamp them.
    var topk_lengths_host = List(length=batch_size, fill=Int32(0))
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
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
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[Q_DEPTH])),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((batch_size, Idx[num_heads], Idx[V_DEPTH])),
    )

    # Row offsets for ragged: [0, 1, 2, ..., batch_size]
    var row_offsets_host = List(length=batch_size + 1, fill=UInt32(0))
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    # Compute and print num_partitions to verify split-K usage.
    comptime sm_count_tc = ctx.default_device_info.sm_count
    var dispatch_scalars_tc = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count_tc // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count_tc)
    var num_partitions_tc = dispatch_scalars_tc[2]
    print(
        "  num_partitions=",
        num_partitions_tc,
        " (split-K",
        "ACTIVE" if num_partitions_tc > 1 else "OFF",
        ")",
    )

    # indices_stride = max topk (allocation stride).
    var indices_stride = max_topk

    print(
        "  Launching MLA decode kernel (sparse, topk clamping)...",
        " max_topk=",
        max_topk,
    )

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
        rope_aware_kv_sparse=True,
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

    # -----------------------------------------------------------------------
    # Verify output
    # -----------------------------------------------------------------------
    var out_host = List(length=out_size, fill=Scalar[q_type](0))
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # FP8 quantization + BF16 accumulation => moderate tolerance.
    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
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
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device
    _ = topk_lengths_host^
    _ = h_indices^
    _ = row_offsets_host^
    _ = cache_lengths_host^
    _ = lookup_table_host^
    _ = out_host^
    _ = ref_host^
    _ = q_host^
    _ = k_sparse_ref^
    _ = k_ref_host^
    _ = k_bf16_host^
    _ = blocks_host^


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # topk=64 out of 257 tokens, batch=1, 16 heads.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_b1_h16_cl256_topk64", 1, 256, ctx, topk=64
            )

            # topk=128 out of 513 tokens, batch=1, 64 heads.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_b1_h64_cl512_topk128", 1, 512, ctx, topk=128
            )

            # topk=64 out of 129 tokens, batch=4, 16 heads.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_b4_h16_cl128_topk64", 4, 128, ctx, topk=64
            )

            # topk=128 out of 1025 tokens, batch=1, 64 heads.
            # split-K should be active.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_b1_h64_cl1024_topk128", 1, 1024, ctx, topk=128
            )

            # topk=256 out of 2049 tokens, batch=1, 64 heads.
            # Guarantees split-K is exercised.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_b1_h64_cl2048_topk256", 1, 2048, ctx, topk=256
            )

            # ---------------------------------------------------------------
            # Blockwise FP8 scaling + variable paged KV cache tests
            # ---------------------------------------------------------------

            # Blockscale: batch=2, variable cache [256, 384], topk=64.
            var cls_2batch: List[Int] = [256, 384]
            run_test_sparse_blockscale[
                DType.bfloat16, DType.float8_e4m3fn, 16, 128
            ]("sparse_blockscale_b2_h16_topk64", cls_2batch, ctx, topk=64)

            # Blockscale: batch=1, 64 heads, topk=128.
            var cls_1batch: List[Int] = [512]
            run_test_sparse_blockscale[
                DType.bfloat16, DType.float8_e4m3fn, 64, 128
            ](
                "sparse_blockscale_b1_h64_cl512_topk128",
                cls_1batch,
                ctx,
                topk=128,
            )

            # Blockscale: batch=3, variable cache [128, 256, 192], topk=64.
            var cls_3batch: List[Int] = [128, 256, 192]
            run_test_sparse_blockscale[
                DType.bfloat16, DType.float8_e4m3fn, 16, 128
            ](
                "sparse_blockscale_b3_h16_var_topk64",
                cls_3batch,
                ctx,
                topk=64,
            )

            # Blockscale: batch=1, 64 heads, long cache, topk=256.
            var cls_splitk: List[Int] = [2048]
            run_test_sparse_blockscale[
                DType.bfloat16, DType.float8_e4m3fn, 64, 128
            ](
                "sparse_blockscale_b1_h64_cl2048_topk256",
                cls_splitk,
                ctx,
                topk=256,
            )

            # ---------------------------------------------------------------
            # Variable per-batch topk tests
            # ---------------------------------------------------------------

            # Variable topk: batch=4, 16 heads.
            # batch 0: cache=256, topk=64
            # batch 1: cache=384, topk=128
            # batch 2: cache=128, topk=32
            # batch 3: cache=512, topk=256
            var vt_cls_4: List[Int] = [256, 384, 128, 512]
            var vt_topk_4: List[Int] = [64, 128, 32, 256]
            run_test_sparse_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_variable_topk_b4_h16",
                vt_cls_4,
                vt_topk_4,
                ctx,
            )

            # Variable topk: batch=2, 64 heads, one large one small.
            # batch 0: cache=1024, topk=256
            # batch 1: cache=128, topk=64
            var vt_cls_2: List[Int] = [1024, 128]
            var vt_topk_2: List[Int] = [256, 64]
            run_test_sparse_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 64
            ](
                "sparse_variable_topk_b2_h64",
                vt_cls_2,
                vt_topk_2,
                ctx,
            )

            # Variable topk: batch=3, 16 heads, all different.
            # batch 0: cache=256, topk=128
            # batch 1: cache=512, topk=64
            # batch 2: cache=192, topk=96
            var vt_cls_3: List[Int] = [256, 512, 192]
            var vt_topk_3: List[Int] = [128, 64, 96]
            run_test_sparse_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_variable_topk_b3_h16",
                vt_cls_3,
                vt_topk_3,
                ctx,
            )

            # ---------------------------------------------------------------
            # Attention sink tests (attn_sink correction)
            # ---------------------------------------------------------------

            # Attn sink: batch=1, 16 heads, small cache (no split-K).
            run_test_sparse_attn_sink[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_attn_sink_b1_h16_cl128_topk64",
                1,
                128,
                ctx,
                topk=64,
            )

            # Attn sink: batch=1, 64 heads, large cache (split-K active).
            run_test_sparse_attn_sink[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_attn_sink_b1_h64_cl1024_topk128",
                1,
                1024,
                ctx,
                topk=128,
            )

            # Attn sink: batch=4, 16 heads, moderate cache.
            run_test_sparse_attn_sink[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # ---------------------------------------------------------------
            # Extra KV (always-attend) tests
            # ---------------------------------------------------------------

            # Extra KV: batch=1, 16 heads, 64 extra tokens (1 block).
            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [128]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_extra_kv[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_extra_kv_b1_h16_topk128_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            # Extra KV: batch=1, 64 heads, 128 extra tokens (2 blocks).
            var ek_cls_2: List[Int] = [512]
            var ek_topk_2: List[Int] = [128]
            var ek_ecls_2: List[Int] = [128]
            var ek_etopk_2: List[Int] = [128]
            run_test_sparse_extra_kv[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_extra_kv_b1_h64_topk128_extra128",
                ek_cls_2,
                ek_topk_2,
                ek_ecls_2,
                ek_etopk_2,
                ctx,
            )

            # Extra KV: batch=4, 16 heads, variable extra_topk per batch.
            var ek_cls_4: List[Int] = [256, 384, 128, 512]
            var ek_topk_4: List[Int] = [64, 128, 64, 128]
            var ek_ecls_4: List[Int] = [64, 128, 64, 192]
            var ek_etopk_4: List[Int] = [64, 64, 32, 128]
            run_test_sparse_extra_kv[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_extra_kv_b4_h16_variable",
                ek_cls_4,
                ek_topk_4,
                ek_ecls_4,
                ek_etopk_4,
                ctx,
            )

            # Extra KV: batch=1, 64 heads, long cache (split-K active).
            var ek_cls_sk: List[Int] = [1024]
            var ek_topk_sk: List[Int] = [256]
            var ek_ecls_sk: List[Int] = [128]
            var ek_etopk_sk: List[Int] = [128]
            run_test_sparse_extra_kv[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_extra_kv_b1_h64_splitk_topk256_extra128",
                ek_cls_sk,
                ek_topk_sk,
                ek_ecls_sk,
                ek_etopk_sk,
                ctx,
            )

            # ---------------------------------------------------------------
            # Topk clamping tests (first execution / small cache)
            # ---------------------------------------------------------------

            # First execution: cache_length=0, actual_tokens=1, topk=64.
            # Kernel must clamp topk from 64 to 1.
            var tc_cls_1: List[Int] = [0]
            var tc_topk_1: List[Int] = [64]
            run_test_sparse_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_topk_clamp_first_exec_b1_h16",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, actual_tokens=6, topk=64.
            # Kernel must clamp topk from 64 to 6.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_topk_clamp_small_cache_b1_h16",
                tc_cls_2,
                tc_topk_2,
                ctx,
            )

            # Mixed batch: cache_length=0 and cache_length=256.
            # batch 0: actual_tokens=1, topk=64 -> clamped to 1
            # batch 1: actual_tokens=257, topk=64 -> no clamping
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_topk_clamp_mixed_b2_h16",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )

            # Mixed batch with 64 heads: cache_length=0 and cache_length=5.
            # batch 0: actual_tokens=1, topk=128 -> clamped to 1
            # batch 1: actual_tokens=6, topk=128 -> clamped to 6
            var tc_cls_4: List[Int] = [0, 5]
            var tc_topk_4: List[Int] = [128, 128]
            run_test_sparse_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 64
            ](
                "sparse_topk_clamp_mixed_b2_h64",
                tc_cls_4,
                tc_topk_4,
                ctx,
            )

            # ---------------------------------------------------------------
            # Variable seq_len tests (q_max_seq_len > 1)
            # ---------------------------------------------------------------

            # seq_len=4: batch=1, 16 heads, moderate cache.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_b1_h16_cl256_topk64_seq4",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )

            # seq_len=8: batch=1, 64 heads, larger cache.
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_b1_h64_cl512_topk128_seq8",
                1,
                512,
                ctx,
                topk=128,
                q_max_seq_len=8,
            )

            # seq_len=4: batch=4, 16 heads (multi-batch + multi-seq).
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_b4_h16_cl128_topk64_seq4",
                4,
                128,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )

            # seq_len=2: batch=1, 64 heads, long cache (split-K active).
            run_test_sparse[.bfloat16, DType.float8_e4m3fn, 64](
                "sparse_b1_h64_cl1024_topk128_seq2",
                1,
                1024,
                ctx,
                topk=128,
                q_max_seq_len=2,
            )
        else:
            pass
