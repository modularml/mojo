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
"""Shared scaffolding for paged-KV MLA prefill tests.

The three SM100 MLA prefill kernel variants (generic, blockscale,
per-token-scale) share the same paged-KV cache layout for K_rope
(``cache_depth = 576`` per token, with the rope window in the last 64
elements). This module provides the boilerplate to:

  - Initialize the paged blocks for K_rope with random data and
    zero-fill the tail past the per-batch ``num_keys``.
  - Build a uniform-batch-size lookup table.
  - Extract the contiguous K_rope slice from the paged storage so the
    naive MHA reference can compute against the same data the kernel
    sees.
  - Provide the runtime list of ``num_keys`` values each test binary
    iterates over (see ``num_keys_to_test``). ``page_size`` is a
    compile-time parameter (one binary per page_size); ``num_keys``
    is purely runtime, so we test the cartesian product of every
    page_size against every value in this list.

The plain bf16 ``flare_mla_prefill`` driver (``run_test_paged_prefill``,
shared by the generic + vhead tests) lives here too, since those two tests
drive the identical kernel path and differ only in the nope/v head-dim
split. The blockscale / per-token-scale tests keep their own per-file
drivers because their kernel calls (FP8 scales / dequant reference) differ.
"""

from std.math import align_up, ceildiv
from std.random import randn
from std.testing import assert_almost_equal

from max.gpu.host import DeviceContext
from std.memory import alloc
from std.utils.index import Index, IndexList

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
from nn.attention.gpu.mha import mha_gpu_naive
from nn.attention.gpu.mla import flare_mla_prefill
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand


# ===-----------------------------------------------------------------------===#
# Constants matching DeepSeek-V2/V3 MLA prefill shapes
# ===-----------------------------------------------------------------------===#

comptime CACHE_DEPTH = 576  # Full MLA cache depth (nope + rope)
comptime ROPE_DEPTH = 64  # Last 64 elements of cache hold rope
comptime NUM_LAYERS = 1  # Single layer per test
comptime KV_NUM_HEADS = 1  # MLA caches a single KV head

# Blockwise FP8 scale granularity: one scale value per 64-element block of
# the head_size axis. With CACHE_DEPTH=576, this yields HEAD_DIM_GRAN=9
# scale blocks per (token, head). The blockscale kernel reads the scale
# for the rope window (head_dim_idx=512..575) at block index 8.
comptime SCALE_BLOCK_SIZE = ROPE_DEPTH  # 64
comptime HEAD_DIM_GRAN = (
    CACHE_DEPTH + SCALE_BLOCK_SIZE - 1
) // SCALE_BLOCK_SIZE  # 9
comptime ROPE_SCALE_BLOCK_IDX = HEAD_DIM_GRAN - 1  # 8 — the block holding rope


# ===-----------------------------------------------------------------------===#
# Paged-KV block layout helpers
# ===-----------------------------------------------------------------------===#


@always_inline
def paged_block_elems(
    total_pages: Int, page_size: Int, head_size: Int = CACHE_DEPTH
) -> Int:
    """Number of scalar elements in the paged block array.

    Block shape (matching ``test_mla_decode_paged_variable.mojo``):
    ``[total_pages, kv_dim2=1, NUM_LAYERS=1, page_size, KV_NUM_HEADS=1,
    head_size]``.
    """
    return total_pages * NUM_LAYERS * page_size * KV_NUM_HEADS * head_size


@always_inline
def page_stride(page_size: Int, head_size: Int = CACHE_DEPTH) -> Int:
    """Per-page element stride in the paged block array."""
    return NUM_LAYERS * page_size * KV_NUM_HEADS * head_size


@always_inline
def token_stride(head_size: Int = CACHE_DEPTH) -> Int:
    """Per-token element stride within a page."""
    return KV_NUM_HEADS * head_size


@always_inline
def lut_max_pages_per_batch(num_keys: Int, page_size: Int) -> Int:
    """LUT row stride (pages-per-batch padded to multiple of 8).

    The 8-page padding matches the SIMD chunk size of the
    ``PagedKVCache.populate`` path.
    """
    return align_up(ceildiv(num_keys, page_size), 8)


# ===-----------------------------------------------------------------------===#
# Random-init + tail-zero-fill for uniform-batch paged blocks
# ===-----------------------------------------------------------------------===#


def fill_paged_blocks_uniform[
    kv_type: DType,
](
    blocks_host: MutPointer[Scalar[kv_type], _],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
    standard_deviation: Float64 = 0.5,
):
    """Fill ``blocks_host`` with random data (bf16 ~ N(0, σ²)) cast
    to ``kv_type``, then zero out tail slots past ``num_keys`` in the
    last page of each batch.

    The randn-then-cast roundtrip keeps the distribution well-defined
    across kv_types (including FP8). Zero-filling the tail makes
    accidental OOB reads contribute negligibly to softmax.

    Use ``standard_deviation=1.0`` (or larger) when ``kv_type`` is an
    FP8 format: smaller stddevs concentrate values near zero, where
    e4m3fn/e5m2 lose precision in the subnormal range. ``0.5`` is fine
    for bf16/half tests and matches the original generic-paged config.

    Each batch is assumed to occupy contiguous pages
    ``[b * num_pages_per_batch, (b+1) * num_pages_per_batch)``.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var total_pages = batch_size * num_pages_per_batch
    var block_elems = paged_block_elems(total_pages, page_size, head_size)

    # Random bf16 → cast to kv_type.
    var blocks_bf16 = alloc[BFloat16](block_elems)
    randn[.bfloat16](
        blocks_bf16,
        block_elems,
        mean=0.0,
        standard_deviation=standard_deviation,
    )
    for i in range(block_elems):
        blocks_host[i] = blocks_bf16[i].cast[kv_type]()
    blocks_bf16.free()

    # Tail zero-fill in last page of each batch.
    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)
    for b in range(batch_size):
        var num_pages_b = num_pages_per_batch
        var valid_in_last = num_keys - (num_pages_b - 1) * page_size
        if valid_in_last == page_size:
            continue
        var last_page = b * num_pages_per_batch + (num_pages_b - 1)
        var base = last_page * pstride + valid_in_last * tstride
        var zero_count = (page_size - valid_in_last) * tstride
        for z in range(zero_count):
            blocks_host[base + z] = 0


# ===-----------------------------------------------------------------------===#
# Lookup-table population for uniform-batch paged blocks
# ===-----------------------------------------------------------------------===#


def fill_uniform_lookup_table(
    lookup_table_host: MutPointer[UInt32, _],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    max_pages_per_batch: Int,
):
    """Populate the lookup table with each batch's pages contiguously.

    For batch ``b``, page ``p`` (in the batch's local page numbering),
    the LUT entry is ``b * num_pages_per_batch + p``. Padding entries
    (between ``num_pages_per_batch`` and ``max_pages_per_batch``) are
    left at zero — the kernel must not read them.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    for i in range(batch_size * max_pages_per_batch):
        lookup_table_host[i] = UInt32(0)
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            lookup_table_host[b * max_pages_per_batch + p] = UInt32(
                b * num_pages_per_batch + p
            )


# ===-----------------------------------------------------------------------===#
# Reference K_rope extraction
# ===-----------------------------------------------------------------------===#


def extract_k_rope_for_batch[
    kv_type: DType,
](
    blocks_host: MutPointer[Scalar[kv_type], MutAnyOrigin],
    out_host: MutPointer[Scalar[kv_type], MutAnyOrigin],
    batch_idx: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
):
    """Copy the rope window (last ``ROPE_DEPTH`` elements of every
    cache token) for batch ``batch_idx`` into ``out_host``.

    ``out_host`` must point to a buffer of at least ``num_keys *
    ROPE_DEPTH`` ``Scalar[kv_type]`` elements. The rope tokens are
    laid out contiguously in token order (matching the
    ``[num_keys, ROPE_DEPTH]`` shape the reference K_ref tile expects).

    Each batch is assumed to occupy contiguous pages
    ``[b * num_pages_per_batch, (b+1) * num_pages_per_batch)``.
    """
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var page_base = batch_idx * num_pages_per_batch
    var rope_offset_in_token = head_size - ROPE_DEPTH

    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)

    for tok in range(num_keys):
        var page_idx = tok // page_size
        var tok_in_page = tok % page_size
        var physical_page = page_base + page_idx

        var src_offset = (
            physical_page * pstride
            + tok_in_page * tstride
            + rope_offset_in_token
        )
        var dst_offset = tok * ROPE_DEPTH
        for d in range(ROPE_DEPTH):
            out_host[dst_offset + d] = blocks_host[src_offset + d]


# ===-----------------------------------------------------------------------===#
# Shared plain-bf16 `flare_mla_prefill` driver (generic + vhead tests)
# ===-----------------------------------------------------------------------===#


def run_test_paged_prefill[
    qkv_type: DType,
    k_rope_type: DType,
    output_type: DType,
    depth: Int,  # Q head dim = nope_depth + ROPE_DEPTH
    num_heads: Int,
    nope_depth: Int,  # K-nope tensor width (== depth - ROPE_DEPTH)
    v_depth: Int = -1,  # V/output width; -1 => v_depth = nope_depth (DeepSeek)
    page_size: Int = 128,
    batch_size: Int = 1,
    diagnostic_bands: Bool = False,
    cache_length: Int = 0,  # pre-existing KV prefix per batch (start_pos)
](seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    """Runs one paged-KV MLA-prefill shape and asserts the output matches
    naive MHA.

    Drives the plain bf16 ``flare_mla_prefill`` path (``KVCacheT`` for
    K_rope, dispatched to ``mla_sm100_prefill_generic`` on B200). Shared by
    the generic-paged test (DeepSeek ``v_head_dim == qk_nope_head_dim``) and
    the CENG-282 vhead test (decoupled ``v_head_dim``).

    ``nope_depth`` is the K-nope / Q@K' contraction width; ``v_depth`` is the
    V / output width. ``v_depth < 0`` (default) means ``v_depth ==
    nope_depth`` (the DeepSeek shape), which reproduces the pre-decoupling
    generic-paged behavior byte-for-byte.

    ``cache_length`` is the pre-existing KV prefix length per batch (the
    kernel's ``start_pos``). The default ``0`` is fresh prefill
    (self-attention over ``[0, seq_len)`` with ``num_keys == seq_len``) and
    is byte-identical to the pre-existing driver behavior. For a cached
    prefix pass ``num_keys > seq_len`` with ``cache_length == num_keys -
    seq_len``: the ``seq_len`` new queries then sit at global positions
    ``[cache_length, num_keys)`` and attend keys ``[0, cache_length + i]``
    for query ``i``. The naive-MHA reference already implements exactly this
    (its dense branch places query ``y`` at ``score_row = y + (num_keys -
    seq_len)`` under ``CausalMask``), so no reference change is needed — only
    the ``cache_lengths`` value below.

    When ``diagnostic_bands`` is False (generic path), prints a single
    ``PASS, max_abs_err`` line and asserts inline. When True (vhead path),
    prints a per-band error breakdown (lower band ``d < nope_depth`` vs upper
    band ``d >= nope_depth``) in a first pass, then asserts in a second pass,
    so the band diagnostic survives even when the assert fires.
    """
    comptime v_dim = nope_depth if v_depth < 0 else v_depth

    comptime if diagnostic_bands:
        print(
            "  [repro] depth(q):",
            depth,
            " nope_depth(Knope):",
            nope_depth,
            " v_depth(V/out):",
            v_dim,
            " num_heads:",
            num_heads,
            " seq_len:",
            seq_len,
            " num_keys:",
            num_keys,
            " page_size:",
            page_size,
        )
    else:
        print(
            "test_mla_prefill_paged",
            " batch_size:",
            batch_size,
            " seq_len:",
            seq_len,
            " num_keys:",
            num_keys,
            " page_size:",
            page_size,
            " qkv_type:",
            qkv_type,
            " k_rope_type:",
            k_rope_type,
        )

    comptime scale = Float32(0.125)

    # ------------------------------------------------------------------
    # Step 1: Allocate ragged Q, K(nope), V on host (random init).
    #   q:        [B*S, H, depth]
    #   k(nope):  [B*Nk, H, nope_depth]
    #   v:        [B*Nk, H, v_dim]
    #   output:   [B*S, H, v_dim]
    # ------------------------------------------------------------------
    var q_size = batch_size * seq_len * num_heads * depth
    var k_size = batch_size * num_keys * num_heads * nope_depth
    var v_size = batch_size * num_keys * num_heads * v_dim
    var o_size = batch_size * seq_len * num_heads * v_dim

    var q_ptr = alloc[Scalar[qkv_type]](q_size)
    var k_ptr = alloc[Scalar[qkv_type]](k_size)
    var v_ptr = alloc[Scalar[qkv_type]](v_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)

    randn[qkv_type](q_ptr, q_size)
    randn[qkv_type](k_ptr, k_size)
    randn[qkv_type](v_ptr, v_size)

    # ------------------------------------------------------------------
    # Step 2: Row-offset tables.
    # ------------------------------------------------------------------
    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    var cache_row_offsets_host = alloc[UInt32](batch_size + 1)
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(i * seq_len)
        cache_row_offsets_host[i] = UInt32(i * num_keys)
    input_row_offsets_host[batch_size] = UInt32(batch_size * seq_len)
    cache_row_offsets_host[batch_size] = UInt32(batch_size * num_keys)

    # ------------------------------------------------------------------
    # Step 3: Paged K_rope blocks + LUT.
    #
    # For the SM100 paged path, ``cache_length(b)`` is the PRE-EXISTING
    # cache length (start_pos in the kernel). For fresh prefill
    # (self-attention with no preceding tokens) it is 0; the kernel then
    # attends to keys ``[0, seq_len)`` where this test places its K_rope
    # data. With a cached prefix (``cache_length > 0``) the kernel attends
    # ``[0, cache_length + i]`` for query ``i`` over the ``num_keys``
    # (== cache_length + seq_len) paged tokens filled here. See
    # ``MLAPositionSummary.get_num_keys_and_start_pos`` in
    # ``mla_prefill_utils.mojo``.
    # ------------------------------------------------------------------
    var num_pages_per_batch = ceildiv(num_keys, page_size)
    var total_pages = batch_size * num_pages_per_batch
    var max_pages_per_batch = lut_max_pages_per_batch(num_keys, page_size)
    var lut_size = batch_size * max_pages_per_batch
    var block_elems = paged_block_elems(total_pages, page_size, CACHE_DEPTH)

    var blocks_host = alloc[Scalar[k_rope_type]](block_elems)
    var cache_lengths_host = alloc[UInt32](batch_size)
    var lookup_table_host = alloc[UInt32](lut_size)

    fill_paged_blocks_uniform[k_rope_type](
        blocks_host, batch_size, num_keys, page_size
    )
    # The naive-MHA reference (Step 9) places the ``seq_len`` queries at the
    # TAIL of the ``num_keys``-token sequence: query ``y`` attends keys
    # ``[0, y + (num_keys - seq_len)]`` under ``CausalMask`` (its start_pos is
    # ``num_keys - seq_len``). The kernel uses ``start_pos = cache_length``, so
    # the two agree ONLY when ``cache_length == num_keys - seq_len``. Any other
    # combination silently compares against a differently-positioned reference
    # and fails with a large, confusing error (this cost real debugging time on
    # CENG-282). Fresh prefill is the ``cache_length == 0, num_keys == seq_len``
    # special case. Unconditional (not ``debug_assert``) so it fires even when
    # asserts are disabled — the case that produced the confusing failure.
    if cache_length + seq_len != num_keys:
        raise Error(
            "run_test_paged_prefill requires cache_length + seq_len == "
            + "num_keys (reference start_pos = num_keys - seq_len). Got "
            + "cache_length="
            + String(cache_length)
            + ", seq_len="
            + String(seq_len)
            + ", num_keys="
            + String(num_keys)
            + "."
        )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_length)
    fill_uniform_lookup_table(
        lookup_table_host,
        batch_size,
        num_keys,
        page_size,
        max_pages_per_batch,
    )

    # ------------------------------------------------------------------
    # Step 4: Device buffers + copy.
    # ------------------------------------------------------------------
    var q_device_buf = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_buf = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_buf = ctx.enqueue_create_buffer[qkv_type](v_size)
    var output_device_buf = ctx.enqueue_create_buffer[output_type](o_size)
    var input_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var blocks_device = ctx.enqueue_create_buffer[k_rope_type](block_elems)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)

    ctx.enqueue_copy(q_device_buf, q_ptr)
    ctx.enqueue_copy(k_device_buf, k_ptr)
    ctx.enqueue_copy(v_device_buf, v_ptr)
    ctx.enqueue_copy(input_ro_buf, input_row_offsets_host)
    ctx.enqueue_copy(cache_ro_buf, cache_row_offsets_host)
    ctx.enqueue_copy(blocks_device, blocks_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 5: TileTensors. K uses nope_depth; V + output use v_dim.
    # ------------------------------------------------------------------
    var q_device = TileTensor(
        q_device_buf,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_device = TileTensor(
        k_device_buf,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[nope_depth])),
    )
    var v_device = TileTensor(
        v_device_buf,
        row_major((batch_size * num_keys, Idx[num_heads], Idx[v_dim])),
    )
    var output_device = TileTensor(
        output_device_buf,
        row_major((batch_size * seq_len, Idx[num_heads], Idx[v_dim])),
    )
    var input_ro_tt = TileTensor(input_ro_buf, row_major(batch_size + 1))
    var cache_ro_tt = TileTensor(cache_ro_buf, row_major(batch_size + 1))

    # ------------------------------------------------------------------
    # Step 6: PagedKVCacheCollection for K_rope.
    # ------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=CACHE_DEPTH, is_mla=True
    )
    var block_shape = IndexList[6](
        total_pages,
        1,  # kv_dim2 = 1 for is_mla
        NUM_LAYERS,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    var blocks_lt = LayoutTensor[k_rope_type, Layout.row_major[6]()](
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
        k_rope_type, kv_params, page_size
    ](
        LayoutTensor[k_rope_type, Layout.row_major[6](), MutAnyOrigin](
            blocks_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[.uint32, lt_layout_2d, ImmutAnyOrigin](
            lookup_table_lt.ptr.as_unsafe_any_origin(),
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(seq_len),  # max_seq_length
        UInt32(num_keys),  # max_cache_length
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # k and v need LayoutTensor form (the paged overload signature).
    var k_lt = k_device.to_layout_tensor()
    var v_lt = v_device.to_layout_tensor()

    # ------------------------------------------------------------------
    # Step 7: Launch the kernel.
    # ------------------------------------------------------------------
    print("  Launching MLA prefill kernel (paged K_rope)...")

    flare_mla_prefill[rank=3](
        output_device,
        q_device,
        k_lt,
        v_lt,
        kv_cache,
        CausalMask(),
        input_ro_tt,
        cache_ro_tt,
        scale,
        ctx,
        q_max_seq_len=seq_len,
    )

    ctx.synchronize()
    print("  Kernel completed (no crash).")

    ctx.enqueue_copy(output_ptr, output_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 8: Build contiguous reference (K_ref, V_ref) of width ``depth``.
    # K_ref = [nope | rope]; V_ref = [realV (v_dim) | zeros].
    # ------------------------------------------------------------------
    var ref_size = batch_size * num_keys * num_heads * depth
    var k_ref_host = alloc[Scalar[qkv_type]](ref_size)
    var v_ref_host = alloc[Scalar[qkv_type]](ref_size)
    var output_ref_host = alloc[Scalar[output_type]](
        batch_size * seq_len * num_heads * depth
    )

    var k_rope_one_batch = alloc[Scalar[k_rope_type]](num_keys * ROPE_DEPTH)

    for b in range(batch_size):
        extract_k_rope_for_batch[k_rope_type](
            blocks_host.as_unsafe_any_origin(),
            k_rope_one_batch.as_unsafe_any_origin(),
            b,
            num_keys,
            page_size,
        )
        for s in range(num_keys):
            for h in range(num_heads):
                var dst_base = (
                    b * num_keys + s
                ) * num_heads * depth + h * depth
                # K_ref: first nope_depth columns from K(nope).
                var k_src_off = (
                    b * num_keys + s
                ) * num_heads * nope_depth + h * nope_depth
                for d in range(nope_depth):
                    k_ref_host[dst_base + d] = k_ptr[k_src_off + d]
                # K_ref: last ROPE_DEPTH columns from extracted rope
                # (broadcast across heads).
                for d in range(ROPE_DEPTH):
                    k_ref_host[dst_base + nope_depth + d] = k_rope_one_batch[
                        s * ROPE_DEPTH + d
                    ].cast[qkv_type]()
                # V_ref: first v_dim columns from real V, rest zero (V has
                # no rope component).
                var v_src_off = (
                    b * num_keys + s
                ) * num_heads * v_dim + h * v_dim
                for d in range(v_dim):
                    v_ref_host[dst_base + d] = v_ptr[v_src_off + d]
                for d in range(depth - v_dim):
                    v_ref_host[dst_base + v_dim + d] = 0

    # ------------------------------------------------------------------
    # Step 9: Naive MHA reference on the contiguous K_ref/V_ref (single
    # ``depth`` for Q/K/V/out).
    # ------------------------------------------------------------------
    var k_ref_device_buf = ctx.enqueue_create_buffer[qkv_type](ref_size)
    var v_ref_device_buf = ctx.enqueue_create_buffer[qkv_type](ref_size)
    var output_ref_device_buf = ctx.enqueue_create_buffer[output_type](
        batch_size * seq_len * num_heads * depth
    )
    ctx.enqueue_copy(k_ref_device_buf, k_ref_host)
    ctx.enqueue_copy(v_ref_device_buf, v_ref_host)

    var q_device_rank4 = TileTensor(
        q_device_buf,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )
    var k_ref_device = TileTensor(
        k_ref_device_buf,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var v_ref_device = TileTensor(
        v_ref_device_buf,
        row_major((batch_size, num_keys, Idx[num_heads], Idx[depth])),
    )
    var output_ref_device = TileTensor(
        output_ref_device_buf,
        row_major((batch_size, seq_len, Idx[num_heads], Idx[depth])),
    )

    var null_valid_length = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](
        None,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(Index(0)),
    )

    var k_ref_operand = LayoutTensorMHAOperand(
        k_ref_device.as_immut().as_unsafe_any_origin()
    )
    var v_ref_operand = LayoutTensorMHAOperand(
        v_ref_device.as_immut().as_unsafe_any_origin()
    )

    mha_gpu_naive[_is_cache_length_accurate=True](
        q_device_rank4.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device.to_layout_tensor(),
        null_valid_length,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,  # group
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ref_host, output_ref_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 10: Compare the first v_dim output columns per head. The kernel
    # writes [total_seq_tokens, num_heads, v_dim]; the reference is rank-4
    # [batch, seq, num_heads, depth].
    # ------------------------------------------------------------------
    comptime atol: Float64 = 2e-2
    comptime rtol: Float64 = 2e-2
    var max_abs_err = Float64(0)

    comptime if diagnostic_bands:
        # First pass: diagnostics (max error + per-band breakdown). Kept
        # separate from the assert so the band info is printed even when the
        # assert fires.
        var max_err_lo = Float64(0)  # columns d < nope_depth
        var max_err_hi = Float64(0)  # columns d >= nope_depth
        var n_mismatch = 0
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(num_heads):
                    for d in range(v_dim):
                        var actual = output_ptr.load(
                            (b * seq_len + s) * num_heads * v_dim
                            + h * v_dim
                            + d
                        ).cast[.float64]()
                        var expect = output_ref_host.load(
                            ((b * seq_len + s) * num_heads + h) * depth + d
                        ).cast[.float64]()
                        var abs_err = abs(actual - expect)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err
                        if d < nope_depth:
                            if abs_err > max_err_lo:
                                max_err_lo = abs_err
                        else:
                            if abs_err > max_err_hi:
                                max_err_hi = abs_err
                        if abs_err > atol:
                            n_mismatch += 1
                            if n_mismatch <= 8:
                                print(
                                    "    mismatch b=",
                                    b,
                                    " s=",
                                    s,
                                    " h=",
                                    h,
                                    " d=",
                                    d,
                                    " actual=",
                                    actual,
                                    " expect=",
                                    expect,
                                )

        print(
            "    max_abs_err:",
            max_abs_err,
            " max_err[d<nope]:",
            max_err_lo,
            " max_err[d>=nope]:",
            max_err_hi,
            " n_mismatch(>atol):",
            n_mismatch,
        )

        # Second pass: assert (raises -> non-zero exit on failure).
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(num_heads):
                    for d in range(v_dim):
                        var actual = output_ptr.load(
                            (b * seq_len + s) * num_heads * v_dim
                            + h * v_dim
                            + d
                        ).cast[.float64]()
                        var expect = output_ref_host.load(
                            ((b * seq_len + s) * num_heads + h) * depth + d
                        ).cast[.float64]()
                        assert_almost_equal(
                            actual, expect, atol=atol, rtol=rtol
                        )
        print("    RESULT: PASS")
    else:
        # Single pass: print mismatches inline + assert.
        for b in range(batch_size):
            for s in range(seq_len):
                for h in range(num_heads):
                    for d in range(v_dim):
                        var actual = output_ptr.load(
                            (b * seq_len + s) * num_heads * v_dim
                            + h * v_dim
                            + d
                        ).cast[.float64]()
                        var expect = output_ref_host.load(
                            ((b * seq_len + s) * num_heads + h) * depth + d
                        ).cast[.float64]()
                        var abs_err = abs(actual - expect)
                        if abs_err > max_abs_err:
                            max_abs_err = abs_err
                        if abs_err > atol:
                            print(
                                "mismatch at b=",
                                b,
                                " s=",
                                s,
                                " h=",
                                h,
                                " d=",
                                d,
                                " actual=",
                                actual,
                                " expect=",
                                expect,
                            )
                        assert_almost_equal(
                            actual, expect, atol=atol, rtol=rtol
                        )

        print("  PASS, max_abs_err:", max_abs_err)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    output_ptr.free()
    k_ref_host.free()
    v_ref_host.free()
    output_ref_host.free()
    k_rope_one_batch.free()
    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()
    cache_row_offsets_host.free()

    _ = q_device_buf
    _ = k_device_buf
    _ = v_device_buf
    _ = output_device_buf
    _ = input_ro_buf
    _ = cache_ro_buf
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = k_ref_device_buf
    _ = v_ref_device_buf
    _ = output_ref_device_buf


# ===-----------------------------------------------------------------------===#
# Blockwise FP8 scales (paged): layout, fill, and reference dequant helpers
# ===-----------------------------------------------------------------------===#
#
# These helpers support tests of MLA prefill kernels that read the FP8
# K_rope cache via a ``PagedKVCacheCollection`` configured with
# ``scale_dtype_=DType.float32, quantization_granularity_=SCALE_BLOCK_SIZE``.
# The scales array is 6D, mirroring the FP8 blocks array but with the
# last axis replaced by ``HEAD_DIM_GRAN`` block-scales:
#
#   shape = [total_pages, kv_dim2=1, NUM_LAYERS=1, page_size,
#            KV_NUM_HEADS=1, HEAD_DIM_GRAN=9]
#
# The blockscale kernel only reads the rope-window scale (block index
# ``ROPE_SCALE_BLOCK_IDX = 8``) but the scales tensor must cover all 9
# blocks because the underlying ``KVCacheT`` operand stores them
# contiguously per-token.


@always_inline
def paged_scale_block_elems(
    total_pages: Int, page_size: Int, head_dim_gran: Int = HEAD_DIM_GRAN
) -> Int:
    """Number of FP32 scale elements in the paged scales array.

    Same shape as ``paged_block_elems`` but with the last axis replaced
    by ``head_dim_gran`` block scales rather than ``head_size``.
    """
    return total_pages * NUM_LAYERS * page_size * KV_NUM_HEADS * head_dim_gran


@always_inline
def scale_page_stride(
    page_size: Int, head_dim_gran: Int = HEAD_DIM_GRAN
) -> Int:
    """Per-page element stride in the paged scales array."""
    return NUM_LAYERS * page_size * KV_NUM_HEADS * head_dim_gran


@always_inline
def scale_token_stride(head_dim_gran: Int = HEAD_DIM_GRAN) -> Int:
    """Per-token element stride within a page in the paged scales array."""
    return KV_NUM_HEADS * head_dim_gran


@always_inline
def _palette_scale(idx: Int) -> Float32:
    """Pick a non-uniform scale from a tight 8-entry palette centered
    around 1.0.

    A tighter range (compared to the decode test's 256x range) keeps
    FP8 quantization noise within tolerance after dequantization, since
    the prefill test compares against an FP8→BF16 dequantized
    reference: ``out = fp8_val * scale``. Wild scales amplify FP8's
    ~5% mantissa error past the 2e-2 tolerance.
    """
    if idx % 8 == 0:
        return 0.5
    if idx % 8 == 1:
        return 0.625
    if idx % 8 == 2:
        return 0.75
    if idx % 8 == 3:
        return 0.875
    if idx % 8 == 4:
        return 1.0
    if idx % 8 == 5:
        return 1.125
    if idx % 8 == 6:
        return 1.25
    return 1.5


def fill_paged_block_scales(
    scales_host: MutPointer[Float32, _],
    batch_size: Int,
    num_keys: Int,
    page_size: Int,
    head_dim_gran: Int = HEAD_DIM_GRAN,
):
    """Fill ``scales_host`` with non-uniform per-(token, block) FP32
    scales drawn from a small palette.

    Using a small palette of order-of-magnitude-1 values (rather than
    ``randn``) keeps the dequantized FP8→BF16 result in a numerically
    well-behaved range for the reference comparison: the kernel does
    ``out = fp8_val * scale`` so wild scales would amplify FP8
    quantization noise past the test's tolerance.

    Tail slots past ``num_keys`` in the last page of each batch are
    filled with neutral 1.0 (matching the kernel's CVT consumer
    behavior, which uses scale=1 for OOB rows — see
    ``cvt_block_fp8_to_bf16_with_scale`` in ``mla_prefill_utils.mojo``).
    """
    var num_pages_per_batch = (num_keys + page_size - 1) // page_size
    var pstride = scale_page_stride(page_size, head_dim_gran)
    var tstride = scale_token_stride(head_dim_gran)

    for b in range(batch_size):
        var page_base = b * num_pages_per_batch
        for pg in range(num_pages_per_batch):
            var physical_page = page_base + pg
            for tok_in_page in range(page_size):
                var tok_global = pg * page_size + tok_in_page
                for blk in range(head_dim_gran):
                    var off = (
                        physical_page * pstride + tok_in_page * tstride + blk
                    )
                    if tok_global < num_keys:
                        # Coprime stride 7 vs palette length 8 so all
                        # entries are exercised even for small token
                        # counts. Vary by both token and block index.
                        scales_host[off] = _palette_scale(tok_global * 7 + blk)
                    else:
                        scales_host[off] = 1.0


def extract_dequantized_k_rope_for_batch[
    fp8_type: DType,
    out_type: DType,
](
    blocks_host: ImmPointer[Scalar[fp8_type], _],
    scales_host: ImmPointer[Float32, _],
    out_host: MutPointer[Scalar[out_type], _],
    batch_idx: Int,
    num_keys: Int,
    page_size: Int,
    head_size: Int = CACHE_DEPTH,
    head_dim_gran: Int = HEAD_DIM_GRAN,
):
    """Extract the rope window for ``batch_idx``, dequantizing per token
    using the matching scale at block index ``ROPE_SCALE_BLOCK_IDX``.

    Equivalent to ``extract_k_rope_for_batch`` followed by
    ``out[t, d] = fp8_val[t, d].cast[float32]() * scale[t,
    ROPE_SCALE_BLOCK_IDX]``, with the result cast to ``out_type``.
    Mirrors the dequantization the blockscale kernel applies via
    ``cvt_block_fp8_to_bf16_with_scale`` (which reads the scale at
    ``head_dim_idx=cache_depth - rope_depth``, i.e. block ``8`` for
    ``cache_depth=576, granularity=64``).

    ``out_host`` must point to a buffer of at least
    ``num_keys * ROPE_DEPTH`` ``Scalar[out_type]`` elements.
    """
    var num_pages_per_batch = (num_keys + page_size - 1) // page_size
    var page_base = batch_idx * num_pages_per_batch
    var rope_offset_in_token = head_size - ROPE_DEPTH

    var pstride = page_stride(page_size, head_size)
    var tstride = token_stride(head_size)
    var spstride = scale_page_stride(page_size, head_dim_gran)
    var ststride = scale_token_stride(head_dim_gran)

    for tok in range(num_keys):
        var page_idx = tok // page_size
        var tok_in_page = tok % page_size
        var physical_page = page_base + page_idx

        var src_offset = (
            physical_page * pstride
            + tok_in_page * tstride
            + rope_offset_in_token
        )
        var scale_offset = (
            physical_page * spstride
            + tok_in_page * ststride
            + ROPE_SCALE_BLOCK_IDX
        )
        var scale_val = scales_host[scale_offset]

        var dst_offset = tok * ROPE_DEPTH
        for d in range(ROPE_DEPTH):
            var fp8_val = blocks_host[src_offset + d].cast[.float32]()
            out_host[dst_offset + d] = (fp8_val * scale_val).cast[out_type]()


# ===-----------------------------------------------------------------------===#
# (page_size × num_keys) cartesian-product test configuration
# ===-----------------------------------------------------------------------===#


def num_keys_to_test() -> List[Int]:
    """Return the list of ``num_keys`` values to test against any
    compile-time ``page_size``.

    Each paged-prefill test binary is built once per ``page_size``
    (a compile-time parameter that flows into
    ``PagedKVCacheCollection[..., page_size]`` and the kernel's
    sub-tile TMA descriptors). Inside ``main`` it iterates over the
    values returned here, so the BUILD rule fans out exactly
    ``len(_MLA_PREFILL_PAGED_PAGE_SIZES)`` binaries per variant rather
    than the cartesian product (compile time dominates over runtime
    cost in these tests).

    The list is the union of every ``num_keys`` previously tested
    across the (page_size, num_keys) configs, so the cartesian product
    of {page_sizes} × {this list} is a strict superset of the
    pre-refactor coverage. It exercises:

      - Aligned baselines: ``num_keys == page_size`` (e.g. 64, 128,
        256), and ``num_keys`` that fills whole BN-sized tiles.
      - Partial-last-tile cases that previously fired the kernel-side
        ``debug_assert`` pre-fix (e.g. (16, 17), (32, 96), (64, 64),
        (16, 100)).
      - Mixed alignment: e.g. (32, 17), (128, 100) — ``num_keys`` not
        a multiple of ``page_size``.
    """
    return [17, 64, 96, 100, 128, 256]
