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

"""Numerical E2E test for MLA_SM100_Decode_Sparse_KV_FP8.

This kernel stores the full 576-byte KV row (nope+rope) as FP8 in HBM,
loaded via a single INT64/SWIZZLE_NONE gather4 (tile_width=72 INT64 =
576 B). The parent kernel (`MLA_SM100_Decode_Sparse`) instead keeps rope
in BF16 (128 bytes) so its rows are 640 bytes.

Layout tested here per KV row:
    [nope: 512 FP8 bytes] [rope: 64 FP8 bytes]  = 576 bytes total

This test covers the full feature matrix supported by the kernel:
  * base (NullMask, no extra features)
  * CausalMask
  * multi-token Q (q_max_seq_len > 1)
  * multi-batch (batch_size > 1)
  * variable per-batch topk (has_variable_topk)
  * attention sinks (has_attn_sink)
  * extra KV (has_extra_kv)
  * topk clamping (small cache with topk > actual_tokens)

For FP8 correctness, every variant:
  1. Casts random BF16 K data to FP8 before storing in the paged cache.
  2. Reads back the FP8 cache and casts to BF16 for the reference.
This ensures the reference sees exactly the same quantized values as the
kernel, so the tolerance measures kernel correctness, not quantization
error.
"""

from std.math import ceildiv, exp
from std.memory import alloc, bitcast
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator, size_of

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
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from std.testing import assert_almost_equal
from std.utils.index import IndexList
from std.utils.numerics import isnan, min_or_neg_inf


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

# ALL-FP8 KV layout: 512 (FP8 nope) + 64 (FP8 rope) = 576 FP8 bytes.
# Unlike test_mla_sparse.mojo (640), rope is ALSO FP8 here.
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH  # 576


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
# Host-side reference: BF16 Q (576) x combined K^T (576) -> P -> O
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
    """Reference MLA with attn_sink correction (natural log domain).

    attn_sink is shape [num_heads_q]. The softmax denominator is adjusted:
      sum_exp += exp(attn_sink[h] - max_s)
    which captures the aggregate contribution of non-selected tokens.
    """
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
                        * k_bf16_ptr[k_base + d].cast[.float64]()
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
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^


def host_reference_varkeys[
    q_type: DType,
](
    q_ptr: Pointer[Scalar[q_type], _],
    k_bf16_ptr: Pointer[Scalar[q_type], _],
    output_ptr: MutPointer[Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA for variable-length per-batch sparse attention.

    K buffer is packed contiguously per batch: k_bf16_ptr contains
    sum_b num_keys_per_batch[b] * depth elements.
    """
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
                        * k_bf16_ptr[k_base + d].cast[.float64]()
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
                    acc += s_buf[k] * k_bf16_ptr[k_base + d].cast[.float64]()
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^
    _ = k_offsets^


# ===-----------------------------------------------------------------------===#
# Core test function (all-FP8 KV, sparse, tensorwise)
# Parametrized over mask type (NullMask / CausalMask) via a `use_causal` flag
# at runtime — the call to flare_mla_decoding picks the right constructor.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    use_causal: Bool = False,
    # Read-once shared-index fold (KERN-3141): build ONE topk list shared by
    # every q position and drive the decode with fold_shared_index=True.
    # False -> per-position distinct scramble (baseline).
    shared_index: Bool = False,
    # Physical gather order of the ONE shared set {0..topk-1} (shared_index
    # only), IDENTICAL across all q positions. 0=identity (slot i == token i,
    # makes the slot-index causal horizon equal logical-token causal),
    # 1=reversed, 2=coprime permutation. NullMask output must be INVARIANT to
    # this order (permutation invariance = the Phase-4 order audit); under
    # CausalMask the kernel's slot-index horizon only equals logical-token
    # causal for order 0 (documented limitation).
    order_mode: Int = 0,
    # Phase 5 seq_len=0: make the LAST batch a 0-length (empty) ragged sequence.
    # Under the fold (grid.y=1, block_idx.y=0) the ragged early-exit fires
    # (block_idx.y >= seq_len) and _pdl_early_exit_all_q writes -inf LSE for
    # every folded slot. Invariant: the empty batch does NOT hang/poison the
    # combine and the non-empty batches stay correct. The empty batch's own
    # output is undefined (skipped in the NaN scan + assert). Default False.
    empty_last_batch: Bool = False,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
    # np-invariance knob (KERN-3217 q1 split-K tuning): >0 forces the split-K
    # partition count via the capturable-graph num_partitions_in override, so a
    # single shape can be verified across np values (default heuristic, the
    # candidate-selected np, and an adversarial over-split). 0 -> use the
    # dispatch heuristic. Drives BOTH the decode grid.z (bs*np) and the combine
    # n_splits, so decode and combine always agree on np.
    forced_np: Int = 0,
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

    # All-FP8 layout: head_size = 576 (V_DEPTH + ROPE_DEPTH).
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

    # Allocate KV cache on host, zero-initialized.
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    # Generate random BF16 data for nope (512) + rope (64) per token.
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # Token stride in the KV cache = head_size = 576 FP8 slots.
    var tok_stride = kv_params.head_size  # 576 FP8 slots

    # Build shuffled page table (so gather4 actually exercises scatter).
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

    # Fill KV cache: nope (512) AND rope (64) both cast BF16 -> FP8.
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

            # Write ALL 576 slots as FP8 (nope + rope both cast).
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    # Reference K: read back FP8 bytes as BF16 so the reference sees
    # exactly the same quantized values as the kernel.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
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
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

    # Q tensor: [batch_size * q_max_seq_len, num_heads, Q_DEPTH] (ragged).
    var q_size = batch_size * q_max_seq_len * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Select topk unique tokens PER QUERY TOKEN via deterministic permutation.
    var selected_tokens = List(length=total_q_tokens * topk, fill=Int(0))
    var mult = _coprime_multiplier(num_keys)
    # Coprime with topk => (i * perm_mult) % topk permutes the shared set
    # {0..topk-1} (order_mode == 2): same SET, shuffled physical gather order.
    var perm_mult = _coprime_multiplier(topk)
    for bi in range(batch_size):
        for s in range(q_max_seq_len):
            var g = bi * q_max_seq_len + s
            for i in range(topk):
                comptime if shared_index:
                    # Index-shared MTP: every q position gets the IDENTICAL
                    # shared list; order_mode picks its physical gather order.
                    # NullMask output must be INVARIANT to order_mode
                    # (permutation invariance). CausalMask matches the host
                    # (logical-token) reference only for order_mode 0, because
                    # the kernel masks by gather-SLOT index — the documented
                    # order limitation (Phase 4). order_mode is a comptime
                    # constant, so this if-chain is trivially folded.
                    if order_mode == 1:
                        selected_tokens[g * topk + i] = topk - 1 - i
                    elif order_mode == 2:
                        selected_tokens[g * topk + i] = (i * perm_mult) % topk
                    else:
                        selected_tokens[g * topk + i] = i
                else:
                    selected_tokens[g * topk + i] = (
                        i * mult + 1 + s
                    ) % num_keys

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
                    k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]

    # For the causal reference we need per-batch virtual token positions
    # for each selected entry. Since each batch uses the same permutation
    # we can reconstruct this from selected_tokens.
    # However the kernel applies causal to logical positions
    # (cache_len + s + 1 relative to the full cache, NOT the sparse set).
    # To match, we use the logical token index `selected_tokens[i]`
    # as the key's position, and clamp each q-token's reach to `cache_len + s + 1`.
    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)

    if use_causal:
        # Build the causal sparse reference using logical token positions.
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
        # (k_sparse_ref is laid out [total_q_tokens, topk, Q_DEPTH]).
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
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for the selected topk tokens.
    #
    # The sparse decode kernel gathers d_indices PER QUERY TOKEN, using the
    # global query-token index g = batch * q_max_seq_len + s as the row, with
    # indices_stride as the per-row stride. So d_indices is laid out as
    # [total_q_tokens, topk]:
    #   d_indices[g * topk + i] = physical_block * PAGE_SIZE + offset
    # where each query token's row uses that token's own selected_tokens set.
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch.
    # -----------------------------------------------------------------------
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
    if empty_last_batch and batch_size > 1:
        # Last batch becomes 0-length: row_offsets[bs] == row_offsets[bs-1].
        row_offsets_host[batch_size] = row_offsets_host[batch_size - 1]
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
        fold_shared_index=shared_index,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads,
        is_fp8_kv=True,
        half_sms=sm_count // 2,
        fold_shared_index=shared_index,
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

    # np-invariance override: force num_partitions when forced_np>0, else let
    # the dispatch heuristic pick. One value drives decode grid.z AND combine
    # n_splits (see flare_mla_decoding.num_partitions_in).
    var _np_ovr = Optional[Int](forced_np) if forced_np > 0 else Optional[Int](
        None
    )

    print(
        "  Launching MLA sparse KV_FP8 decode kernel...",
        " topk=",
        topk,
        " num_keys=",
        num_keys,
        " forced_np=",
        forced_np,
    )

    comptime if use_causal:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
            fold_shared_index=shared_index,
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
            num_partitions_in=_np_ovr,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
            fold_shared_index=shared_index,
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
            num_partitions_in=_np_ovr,
        )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output: max abs error must be < 5e-2 and zero NaN allowed.
    # -----------------------------------------------------------------------
    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
    var nan_count = 0
    var total_checked = 0
    for b in range(batch_size):
        if empty_last_batch and b == batch_size - 1:
            continue  # empty batch: output undefined (fold ragged early-exit)
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
                    if err > 1e-1:
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

    # Run asserts only after NaN scan (so we see all NaNs before failing).
    for b in range(batch_size):
        if empty_last_batch and b == batch_size - 1:
            continue  # empty batch: output undefined (fold ragged early-exit)
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
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Variable-topk sparse test (has_variable_topk=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_variable_topk[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Variable-topk per-batch with all-FP8 KV (576-byte row).

    Per-batch cache lengths and topk counts. indices_stride = max(topk).
    """
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
    print("  indices_stride (max topk)=", max_topk)

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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
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

    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # FP8-roundtripped reference K.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

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

            # Write all 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()

        k_offset += nk * Q_DEPTH

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # d_indices: [batch_size * max_topk] padded with zeros beyond each
    # batch's actual topk.
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
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
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

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = max_topk
    print("  Launching MLA sparse KV_FP8 (variable topk)...")

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


# ===-----------------------------------------------------------------------===#
# Attention sink sparse test (has_attn_sink=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_attn_sink[
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
    """All-FP8 sparse MLA decode with attn_sink correction.

    attn_sink shape is [num_heads_q] (NOT batch_size) per the MOGG fix.
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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

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
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    # FP8-roundtripped reference K.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
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
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

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
                k_sparse_ref[dst + d] = k_ref_host[src + d]

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

    # Uploads.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
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
    print("  Launching MLA sparse KV_FP8 (attn_sink)...")

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
        attn_sink_ptr=rebind[MutPointer[Float32, origin=MutAnyOrigin]](
            attn_sink_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
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

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

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
# Extra KV sparse test (has_extra_kv=True)
# Both the original and extra KV caches use the same 576-byte all-FP8 layout.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_extra_kv[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Two paged KV caches (main + always-attend). Both all-FP8, 576 B/row."""
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

    # --- ORIGINAL cache ----------------------------------------------------
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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
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
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

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
            # All 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()
        k_offset += nk * Q_DEPTH

    # --- EXTRA cache -------------------------------------------------------
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

    var extra_blocks_host = ctx.enqueue_create_host_buffer[kv_type](
        extra_block_elems
    )
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[kv_type](0)

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
    var extra_k_bf16_host = ctx.enqueue_create_host_buffer[q_type](
        extra_total_k_elems
    )
    randn(
        extra_k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var extra_k_ref_host = ctx.enqueue_create_host_buffer[q_type](
        extra_total_k_elems
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
            # All 576 slots as FP8 — same layout as the main cache.
            for d in range(Q_DEPTH):
                var fp8_val = extra_k_bf16_host[k_base + d].cast[kv_type]()
                extra_blocks_host[base + d] = fp8_val
                extra_k_ref_host[k_base + d] = fp8_val.cast[q_type]()
        ek_offset += enk * Q_DEPTH

    # Q
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Select indices for both caches; build combined reference.
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
    var total_combined_ref_elems = 0
    for bi in range(batch_size):
        total_combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * Q_DEPTH
    var k_combined_ref = ctx.enqueue_create_host_buffer[q_type](
        total_combined_ref_elems
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
                k_combined_ref[dst_base + d] = k_ref_host[src_base + d]
        combined_ref_offset += topk_bi * Q_DEPTH

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
            var src_base = ek_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = extra_k_ref_host[src_base + d]
        combined_ref_offset += extra_topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH
        ek_offset_src += enk * Q_DEPTH

    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

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
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

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

    # Build KV collections (original + extra).
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse KV_FP8 (extra KV)...")

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

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

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
# Topk-clamping sparse test (small caches with topk > actual_tokens)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_topk_clamping[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Topk clamping with all-FP8 KV layout.

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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
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
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

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
            # All 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()
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
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
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

    var topk_lengths_host = ctx.enqueue_create_host_buffer[.int32](batch_size)
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])  # UNCLAMPED
    var topk_lengths_device = ctx.enqueue_create_buffer[.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

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
        LayoutTensor[mut=False, .uint32, cl_layout, _](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[mut=False, .uint32, lt_layout_2d, _](
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
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = max_topk
    print("  Launching MLA sparse KV_FP8 (topk clamping)...")

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

            # A split with no compiled combine kernel must fail the launch
            # instead of skipping the reduction. The message is asserted, not
            # just the raise, because an unreduced output also fails the value
            # comparison. 5 stays unbucketed if the table grows.
            var unbucketed_raised = False
            try:
                run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                    "unbucketed_split_must_raise",
                    2,
                    2048,
                    ctx,
                    topk=2048,
                    q_max_seq_len=1,
                    forced_np=5,
                )
            except e:
                if "no compiled combine kernel" in String(e):
                    unbucketed_raised = True
                else:
                    raise Error(
                        String(
                            "a split outside the bucket table failed for the"
                            " wrong reason: "
                        )
                        + String(e)
                    )
            if not unbucketed_raised:
                raise Error(
                    "a split outside the bucket table was accepted, so the"
                    " split-K reduction can still be skipped silently"
                )

            # =====================================================
            # KERN-3217 q1 split-K tuning: np-invariance + zero-work coverage.
            # The dispatch change relaxes the split-K page floor for the
            # q_len=1, batch_size<=8, sparse FP8 (NullMask, no extra_kv /
            # variable_topk / attn_sink) path so np tracks the effective
            # (clamped) KV page count.
            # These cases prove the split-K decode + combine produce the
            # SAME (reference-matching) output across np: the default heuristic,
            # the candidate-selected np, and an adversarial over-split forcing
            # zero-work splits. forced_np drives BOTH decode grid.z and combine
            # n_splits, so decode/combine always agree on np. (Read the ACTUAL
            # launched np from nsys grid.z; the printed num_partitions is the
            # cache-based heuristic value, which does not reflect the sparse
            # launch np.)
            # =====================================================
            # --- effective-2048 domain (cache=2048, topk=2048 -> 16 pages):
            #     candidate heuristic selects np=16 (one page per split). ---
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs2_default",
                2,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=0,
            )
            # The one table entry no shipped path had launched before.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs2_np3",
                2,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=3,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs2_np4",
                2,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=4,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs2_np8",
                2,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=8,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs2_np16",
                2,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=16,
            )
            # Primary reviewer batch (bs=8): default (candidate np=16) + np=16.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs8_default",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=0,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff2048_bs8_np16",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=16,
            )
            # --- effective-1024 domain (cache=1024, topk=1024 -> 8 pages):
            #     forced-np proxy shape (topk != 2048, so the tuning gate is
            #     false and the heuristic keeps the old policy); np=8 is the
            #     one-page-per-split value forced here for invariance. ---
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff1024_bs2_default",
                2,
                1024,
                ctx,
                topk=1024,
                q_max_seq_len=1,
                forced_np=0,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff1024_bs2_np8",
                2,
                1024,
                ctx,
                topk=1024,
                q_max_seq_len=1,
                forced_np=8,
            )
            # ADVERSARIAL over-split: 8 pages forced to np=16 -> 8 zero-work
            # splits. Must stay reference-correct with zero NaN (validates the
            # num_keys_this_split==0 early-exit + combine -inf partition
            # handling). The heuristic will NOT select this (np is capped at the
            # page count); coverage is retained via the explicit override.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_eff1024_bs2_np16_ADVERSARIAL_oversplit",
                2,
                1024,
                ctx,
                topk=1024,
                q_max_seq_len=1,
                forced_np=16,
            )
            # --- 9-page domain (cache=1152, topk=1152 -> 9 pages): matches the
            #     CLAMPED work of the in-scope cache=1024/topk=2048 shape, for
            #     which the tuned dispatch selects np=9. Here topk != 2048, so
            #     the gate is false; np=9 is forced for invariance coverage. ---
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_9page_bs2_default",
                2,
                1152,
                ctx,
                topk=1152,
                q_max_seq_len=1,
                forced_np=0,
            )
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_9page_bs2_np16_ADVERSARIAL_oversplit",
                2,
                1152,
                ctx,
                topk=1152,
                q_max_seq_len=1,
                forced_np=16,
            )
            # --- exact clamped production shape (cache=1024, topk=2048):
            #     the KERN-3217 tuning gate is TRUE here, so the automatic
            #     dispatch clamps the split length to min(2048, 1024+1) ->
            #     9 pages and selects np=9 itself (forced_np=0). ---
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 8](
                "q1_np_inv_topk2048_cache1024_bs2_default",
                2,
                1024,
                ctx,
                topk=2048,
                q_max_seq_len=1,
                forced_np=0,
            )

            # =====================================================
            # Read-once shared-index MTP fold (KERN-3141), shared_index=True.
            # All q positions share ONE identity-ordered topk list; the fold
            # gathers it ONCE (grid.y=1) and every BM row (q_len*nqh=48) attends
            # the single pass. Driven by the fold_shared_index comptime param
            # (no -D). These exercise the fold branches (NullMask + CausalMask);
            # Phase 4 adds the order-correctness matrix (reversed / random /
            # score-order / permuted-physical) and the different-set negative.
            # =====================================================
            # NullMask, single tile: bs1, nqh8, q6, cl256, topk64 (48<=BM=64).
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b1_h8_cl256_topk64_seq6",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            # Multi-tile shared list: bs2, topk160 = 2.5 tiles.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b2_h8_cl1024_topk160_seq6",
                2,
                1024,
                ctx,
                topk=160,
                q_max_seq_len=6,
            )
            # CausalMask tail: cl64, topk70=num_keys -> identity list includes
            # the draft tokens (indices 64..69); position 5 sees its draft
            # tokens, position 0 does not (per-position causal horizon). This
            # instantiates the CausalMask fold branch.
            run_test_sparse_kv_fp8[
                DType.bfloat16,
                DType.float8_e4m3fn,
                8,
                use_causal=True,
                shared_index=True,
            ](
                "shared_fold_causal_tail_b1_h8_cl64_topk70_seq6",
                1,
                64,
                ctx,
                topk=70,
                q_max_seq_len=6,
            )
            # --- Phase 4 order audit: NullMask permutation invariance. ---
            # Same shared SET as the identity cases, but a reversed / coprime-
            # permuted physical gather order. NullMask has no causal horizon, so
            # every folded row attends the whole shared set: output MUST be
            # invariant to gather order. Passing against the (order-agnostic)
            # NullMask reference proves the read-once fold is permutation
            # invariant where the attention semantics are order invariant.
            run_test_sparse_kv_fp8[
                DType.bfloat16,
                DType.float8_e4m3fn,
                8,
                shared_index=True,
                order_mode=1,
            ](
                "shared_fold_null_reversed_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            run_test_sparse_kv_fp8[
                DType.bfloat16,
                DType.float8_e4m3fn,
                8,
                shared_index=True,
                order_mode=2,
            ](
                "shared_fold_null_permuted_b2_h8_cl512_topk160_seq6",
                2,
                512,
                ctx,
                topk=160,
                q_max_seq_len=6,
            )
            # --- Phase 4 negative: DIFFERENT per-position sets. The fold
            # precondition does NOT hold, so the production gate
            # (index_share = reuse_prev_topk and not skip_topk) is False and the
            # UNFOLDED baseline runs (shared_index=False here). Each of the 6 q
            # positions keeps its own distinct scrambled list; matching the
            # per-position reference proves one folded row never silently
            # represents six different sets (the fold is NOT selected here).
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "neg_distinct_sets_unfolded_b2_h8_cl512_topk64_seq6",
                2,
                512,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )
            # The geometry the cost model selects in production, which no
            # other case covers: multi-token and an odd split.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "unfolded_np3_b2_h8_cl512_topk64_seq6",
                2,
                512,
                ctx,
                topk=64,
                q_max_seq_len=6,
                forced_np=3,
            )
            # --- Phase 5 shape matrix (NullMask, shared_index): topk tile
            # boundaries, batch sizes, and the production split-K shape. ---
            # topk=40: partial first tile (< BN_QK=64).
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b3_h8_cl256_topk40_seq6",
                3,
                256,
                ctx,
                topk=40,
                q_max_seq_len=6,
            )
            # topk=63/64/65: straddle the single-tile boundary (partial-last-tile
            # off-by-one coverage).
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b3_h8_cl256_topk63_seq6",
                3,
                256,
                ctx,
                topk=63,
                q_max_seq_len=6,
            )
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b3_h8_cl256_topk65_seq6",
                3,
                256,
                ctx,
                topk=65,
                q_max_seq_len=6,
            )
            # topk=1024 + cache=2048: multi-page, split-K active.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b3_h8_cl2048_topk1024_seq6",
                3,
                2048,
                ctx,
                topk=1024,
                q_max_seq_len=6,
            )
            # Every shape above sits on a page or tile boundary, so a length
            # that divides evenly is the only one either path has run. Prime
            # cache and topk put the partial last page, the partial last tile
            # and the split remainder all off their boundaries at once.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_b2_h8_cl1021_topk101_seq6",
                2,
                1021,
                ctx,
                topk=101,
                q_max_seq_len=6,
            )
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=False
            ](
                "unfolded_b3_h8_cl1279_topk257_seq5",
                3,
                1279,
                ctx,
                topk=257,
                q_max_seq_len=5,
            )
            # Production shape: bs8, cache=2048, topk=2048 (every token), the
            # benchmark shape; split-K over the true 2048 domain (fold relaxes
            # the page floor -> the many-way split the Phase-7 bench measures).
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "shared_fold_prod_b8_h8_cl2048_topk2048_seq6",
                8,
                2048,
                ctx,
                topk=2048,
                q_max_seq_len=6,
            )
            # --- Phase 5 unsupported-ragged FALLBACK (dispatch-negative). ---
            # q_len=1 is BELOW MIN_FOLD_Q=2, so the dispatch fold-selection loop
            # `for n in [2,8]` never matches even with shared_index requested ->
            # the UNFOLDED baseline runs (grid.y=1 for a single position). This
            # is exactly the production disjointness (Phase 8: index_share is
            # only True at q_len=1, where the fold cannot be selected). Output
            # matching the per-position reference proves the fold is NOT selected
            # for an unsupported q shape; a wrong impl that folded q=1 would
            # mis-pack. (An unsupported ragged shape falls back the same way.)
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 8, shared_index=True
            ](
                "unsupported_q1_fallback_b3_h8_cl256_topk64_seq1",
                3,
                256,
                ctx,
                topk=64,
                q_max_seq_len=1,
            )
            # --- Phase 5 seq_len=0: last batch empty under the fold. ---
            # bs2, q6, shared-index fold; batch 1 has 0 tokens. The fold's
            # ragged early-exit + _pdl_early_exit_all_q must write -inf LSE for
            # the empty batch's folded slots WITHOUT hanging/poisoning the
            # combine; batch 0 (full, 6 positions) stays correct. A missing
            # all-q early-exit would leave uninitialized LSE -> combine reads
            # garbage -> CUDA_ERROR_ILLEGAL_ADDRESS / wrong batch-0 output.
            run_test_sparse_kv_fp8[
                DType.bfloat16,
                DType.float8_e4m3fn,
                8,
                shared_index=True,
                empty_last_batch=True,
            ](
                "shared_fold_seqlen0_b2_h8_cl256_topk64_seq6",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=6,
            )

            # =====================================================
            # Base variants: NullMask, no feature flags.
            # =====================================================

            # Minimal config: bs=1, h=16, cl=128, topk=8.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_min_b1_h16_cl128_topk8",
                1,
                128,
                ctx,
                topk=8,
            )

            # bs=1, h=16, cl=512, topk=64.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Production-ish: bs=1, h=128, cl=512, topk=64.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 128](
                "sparse_kv_fp8_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Larger cache: bs=1, h=128, cl=2048, topk=64 (split-K likely).
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 128](
                "sparse_kv_fp8_b1_h128_cl2048_topk64",
                1,
                2048,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-batch (batch_size > 1). Exercises per-batch indexing.
            # =====================================================

            # bs=2, h=16, cl=128, topk=64.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b2_h16_cl128_topk64",
                2,
                128,
                ctx,
                topk=64,
            )

            # bs=4, h=16, cl=256, topk=64.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-token query (q_max_seq_len > 1) — speculative decode.
            # =====================================================

            # q_max_seq_len=2.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl256_topk64_seq2",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=2,
            )

            # q_max_seq_len=4.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl256_topk64_seq4",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )

            # q_max_seq_len=8, multi-batch.
            run_test_sparse_kv_fp8[.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=8,
            )

            # =====================================================
            # CausalMask variant.
            # =====================================================

            # Causal, single-token.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 16, use_causal=True
            ](
                "sparse_kv_fp8_causal_b1_h16_cl256_topk64",
                1,
                256,
                ctx,
                topk=64,
            )

            # Causal + multi-token: we intentionally skip this combination.
            # For sparse MLA decode the kernel's causal mask interacts with
            # logical token positions inside the full cache, but the sparse
            # indices expose only a subset of those positions. The exact
            # mapping from (q_seq, sparse_slot) -> causal validity isn't
            # straightforward to reference on the host without duplicating
            # kernel internals; the parent test_mla_sparse.mojo also does
            # not cover causal+multi-seq directly. The seq_len=1 causal
            # case above exercises the mask wiring end-to-end.

            # =====================================================
            # Variable per-batch topk (has_variable_topk=True).
            # =====================================================

            # bs=2, variable cache + variable topk.
            var vt_cls_2: List[Int] = [256, 128]
            var vt_topk_2: List[Int] = [64, 32]
            run_test_sparse_kv_fp8_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_variable_topk_b2_h16",
                vt_cls_2,
                vt_topk_2,
                ctx,
            )

            # bs=4, variable cache + variable topk.
            var vt_cls_4: List[Int] = [256, 384, 128, 512]
            var vt_topk_4: List[Int] = [64, 128, 32, 64]
            run_test_sparse_kv_fp8_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_variable_topk_b4_h16",
                vt_cls_4,
                vt_topk_4,
                ctx,
            )

            # =====================================================
            # Attention sink (has_attn_sink=True).
            # =====================================================

            # attn_sink, small cache, 16 heads.
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_attn_sink_b1_h16_cl128_topk64",
                1,
                128,
                ctx,
                topk=64,
            )

            # attn_sink, production-ish (128 heads).
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 128
            ](
                "sparse_kv_fp8_attn_sink_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # attn_sink, multi-batch.
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Extra KV (has_extra_kv=True).
            # =====================================================

            # bs=1, 16 heads, 64 extra tokens.
            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [64]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_kv_fp8_extra_kv[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_extra_kv_b1_h16_topk64_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            # bs=2, 16 heads, variable extra.
            var ek_cls_2: List[Int] = [256, 384]
            var ek_topk_2: List[Int] = [64, 64]
            var ek_ecls_2: List[Int] = [64, 128]
            var ek_etopk_2: List[Int] = [64, 64]
            run_test_sparse_kv_fp8_extra_kv[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_extra_kv_b2_h16_variable",
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
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_first_exec_b1_h16",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, actual=6, topk=64 -> clamp to 6.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_small_cache_b1_h16",
                tc_cls_2,
                tc_topk_2,
                ctx,
            )

            # Mixed batch: cl=0 and cl=256.
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_mixed_b2_h16",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )
