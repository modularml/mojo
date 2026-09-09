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
"""Test MLA prefill with a paged BF16 K_rope cache (per-token-scale kernel).

Companion to ``test_mla_prefill_blockscale_paged.mojo`` and
``test_mla_prefill_generic_paged.mojo``: exercises the per-token-scale
SM100 MLA prefill kernel (``mla_prefill_per_token_scale.mojo``) under
the same partial-page configs.

Routing to ``mla_sm100_prefill_per_token_scale`` happens through the
NEW ``flare_mla_prefill`` overload added alongside the contiguous
per-token-scale entrypoint at ``mla.mojo:1998``: it takes Q_nope,
Q_rope, Q_scale, K, K_scale, V as TileTensors plus a paged
``KVCacheT`` for K_rope (no integrated FP32 block-scales — per-token-
scale uses contiguous K_scale instead). The cache stays in BF16
divided by per-token K_scale (matching the contiguous test
``test_mla_prefill_per_token_scale_fp8.mojo`` quantization model).

Why this test exists
====================

The existing ``test_mla_prefill_per_token_scale_fp8.mojo`` covers only
the contiguous K_rope path (``LayoutTensorMHAOperand``) and so does
not exercise the kernel's paged sub-tile TMA loops at lines
684/817/968/1117 (K_rope) and 709/838/994/1136 (k_scale). The
kernel-side ``debug_assert`` placed in those loops fires whenever a
partial-tile config tries to issue a TMA past the sequence end (the
silent partial-page bug — see
``docs/plans/sorted-sauteeing-snowglobe.md`` and
``test_mla_prefill_generic_paged.mojo`` for the full discovery
history).

In this test K (and therefore V and k_scale's row mapping) is
contiguous, so the k_scale sub-tile loop's ``num_kv_pages`` is 1 and
those asserts are not reachable in this test setup. The K_rope sub-
tile loop's ``num_rope_pages`` IS multi-valued whenever
``KRopeType.page_size < BN`` (i.e. ``page_size < 128``) — those four
sites are where the asserts fire pre-fix.

Configs whose ``num_rope_pages`` loop hits a fully-OOB sub-page (and
therefore fail pre-fix): ``ps64_nk64``, ``ps32_nk96``,
``ps16_nk17``, ``ps16_nk100``. Configs where every issued sub-page
has a valid first row (and therefore pass both pre- and post-fix):
``ps256_nk256``, ``ps128_nk128``, ``ps128_nk256``, ``ps128_nk100``,
``ps64_nk256``, ``ps64_nk100``, ``ps32_nk100``.
"""

from std.math import ceildiv
from std.random import randn, seed
from std.sys import get_defined_int

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
from nn.attention.mha_mask import CausalMask
from nn.attention.mha_operand import LayoutTensorMHAOperand
from nn.attention.gpu.mla import flare_mla_prefill
from std.testing import assert_almost_equal
from max.gpu.host.info import _is_sm10x_gpu
from std.utils.index import Index, IndexList

from _paged_prefill_test_utils import (
    CACHE_DEPTH,
    KV_NUM_HEADS,
    NUM_LAYERS,
    ROPE_DEPTH,
    fill_uniform_lookup_table,
    lut_max_pages_per_batch,
    num_keys_to_test,
    page_stride,
    paged_block_elems,
    token_stride,
)


# ===-----------------------------------------------------------------------===#
# Compile-time parameterisation. ``num_keys`` is iterated at runtime; only
# ``page_size`` requires a separate compilation since
# ``PagedKVCacheCollection[..., page_size]`` is parameterised on it.
# ===-----------------------------------------------------------------------===#

comptime PAGE_SIZE = get_defined_int["page_size", 256]()


# ===-----------------------------------------------------------------------===#
# Test scaffolding
# ===-----------------------------------------------------------------------===#


def run_test_paged_prefill_per_token_scale[
    qkv_type: DType,
    rope_type: DType,
    scale_type: DType,
    output_type: DType,
    depth: Int,
    num_heads: Int,
    kv_depth: Int,
    page_size: Int,
    batch_size: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext) raises:
    print(
        "test_mla_prefill_per_token_scale_paged",
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
        " rope_type:",
        rope_type,
        " scale_type:",
        scale_type,
    )

    var scale = Float32(0.5)
    var scale_factor = BFloat16(0.125)

    # The per-token-scale kernel's q_scale TMA descriptor
    # (`per_token_scale.mojo:1421`) and the analogous k_scale TMA tile
    # builder both require:
    #   - the descriptor's flat extent to be 4-divisible (FP32 → 16-byte
    #     inner-dim alignment);
    #   - the per-batch col_offset (= batch * stride) to be 4-aligned.
    # When `seq_len` or `num_keys` is not a multiple of 4 (e.g.
    # ``ps16_nk17``), pad each batch's storage stride up to a
    # 4-multiple so both constraints hold simultaneously, while keeping
    # the test's function signature parameterised by the LOGICAL
    # `seq_len` / `num_keys` (so the BUILD config and print labels stay
    # at their nominal values). Real data is filled in the first
    # `seq_len` / `num_keys` slots per batch; trailing padding is zero
    # for K/V/Q (so they don't contribute to attention) and 1.0 for
    # q_scale / k_scale (neutral scale).
    var seq_len_padded = ((seq_len + 3) // 4) * 4
    var num_keys_padded = ((num_keys + 3) // 4) * 4

    # ------------------------------------------------------------------
    # Step 1: Allocate ragged Q (split into nope+rope), K, V (all
    # contiguous, FP8 Q_nope/K/V, BF16 Q_rope), plus FP32 per-token
    # Q_scale and K_scale. K_rope (cache) lives in a paged
    # KVCacheCollection — the ONLY paged operand in this test. Generate
    # the BF16 source data first, then quantize per the per-token-scale
    # kernel's model. Storage strides use the padded values; data
    # fills only `seq_len` / `num_keys` valid entries per batch.
    # ------------------------------------------------------------------
    var q_size = batch_size * seq_len_padded * num_heads * depth
    var q_nope_size = batch_size * seq_len_padded * num_heads * kv_depth
    var q_rope_size = (
        batch_size * seq_len_padded * num_heads * (depth - kv_depth)
    )
    var q_scale_size = batch_size * seq_len_padded
    var k_size = batch_size * num_keys_padded * num_heads * kv_depth
    var k_scale_size = batch_size * num_keys_padded
    var v_size = k_size
    var o_size = batch_size * seq_len_padded * num_heads * kv_depth
    # cache_bf16_ptr is a host-only buffer for the reference K_ref/V_ref
    # build path. It mirrors the paged K_rope blocks but laid out
    # contiguously per (batch, num_keys, head, depth). Use logical
    # num_keys here since the reference path indexes by logical row.
    var cache_size_per_batch = num_keys * KV_NUM_HEADS * CACHE_DEPTH
    var cache_size = batch_size * cache_size_per_batch

    var q_nope_ptr = alloc[Scalar[qkv_type]](q_nope_size)
    var q_rope_ptr = alloc[Scalar[rope_type]](q_rope_size)
    var q_scale_ptr = alloc[Scalar[scale_type]](q_scale_size)
    var k_ptr = alloc[Scalar[qkv_type]](k_size)
    var k_scale_ptr = alloc[Scalar[scale_type]](k_scale_size)
    var v_ptr = alloc[Scalar[qkv_type]](v_size)
    var output_ptr = alloc[Scalar[output_type]](o_size)

    # Zero-init the padded buffers so trailing rows past the logical
    # `seq_len` / `num_keys` per batch contribute nothing to attention.
    # (Real data overwrites the first `seq_len` / `num_keys` rows below.)
    for i in range(q_nope_size):
        q_nope_ptr[i] = Scalar[qkv_type](0)
    for i in range(q_rope_size):
        q_rope_ptr[i] = Scalar[rope_type](0)
    for i in range(q_scale_size):
        q_scale_ptr[i] = Float32(1.0).cast[scale_type]()
    for i in range(k_size):
        k_ptr[i] = Scalar[qkv_type](0)
    for i in range(k_scale_size):
        k_scale_ptr[i] = Float32(1.0).cast[scale_type]()
    for i in range(v_size):
        v_ptr[i] = Scalar[qkv_type](0)

    # BF16 source data, scaled down by 0.125 to keep the FP8 quant range
    # well-conditioned. Source buffers are sized for the LOGICAL
    # `seq_len` / `num_keys` since they're only used to seed the
    # quantization loops below (which iterate over the logical extents).
    var q_bf16_size = batch_size * seq_len * num_heads * depth
    var k_bf16_size = batch_size * num_keys * num_heads * kv_depth
    var v_bf16_size = k_bf16_size
    var q_bf16_ptr = alloc[BFloat16](q_bf16_size)
    var k_bf16_ptr = alloc[BFloat16](k_bf16_size)
    var v_bf16_ptr = alloc[BFloat16](v_bf16_size)
    var cache_bf16_ptr = alloc[BFloat16](cache_size)

    randn[.bfloat16](q_bf16_ptr, q_bf16_size)
    randn[.bfloat16](k_bf16_ptr, k_bf16_size)
    randn[.bfloat16](v_bf16_ptr, v_bf16_size)
    randn[.bfloat16](cache_bf16_ptr, cache_size)
    for i in range(q_bf16_size):
        q_bf16_ptr[i] *= scale_factor
    for i in range(k_bf16_size):
        k_bf16_ptr[i] *= scale_factor
    for i in range(v_bf16_size):
        v_bf16_ptr[i] *= scale_factor
    for i in range(cache_size):
        cache_bf16_ptr[i] *= scale_factor

    # ------------------------------------------------------------------
    # Step 2: Quantize Q (per-token Q_scale), split into Q_nope (FP8)
    # and Q_rope (BF16, scaled); quantize K (per-token K_scale → FP8);
    # cast V directly to FP8 (no V scale in PTS). Source data is read
    # from the LOGICAL-stride bf16 buffers; quantized output is written
    # at the PADDED-stride per-batch offsets so the kernel's TMA col-
    # offsets (= padded stride * batch_idx) are 4-aligned.
    # ------------------------------------------------------------------
    # Q_scale per token: max(|q_bf16|) / 448 across heads × depth.
    for b in range(batch_size):
        for s in range(seq_len):
            var q_max = Float32(-1e10)
            for h in range(num_heads):
                for d in range(depth):
                    var off = ((b * seq_len + s) * num_heads + h) * depth + d
                    var q_abs = abs(q_bf16_ptr[off]).cast[.float32]()
                    if q_abs > q_max:
                        q_max = q_abs
            q_scale_ptr[b * seq_len_padded + s] = max(
                q_max / Float32(448), Float32(1e-10)
            ).cast[scale_type]()

    # Split Q into FP8 nope + BF16 rope (both divided by Q_scale).
    for b in range(batch_size):
        for s in range(seq_len):
            var qs = q_scale_ptr[b * seq_len_padded + s].cast[.bfloat16]()
            for h in range(num_heads):
                for d in range(depth):
                    var src_off = (
                        (b * seq_len + s) * num_heads + h
                    ) * depth + d
                    var v_bf16 = q_bf16_ptr[src_off]
                    if d < kv_depth:
                        var dst_off = (
                            (b * seq_len_padded + s) * num_heads + h
                        ) * kv_depth + d
                        q_nope_ptr[dst_off] = (v_bf16 / qs).cast[qkv_type]()
                    else:
                        var dst_off = (
                            (b * seq_len_padded + s) * num_heads + h
                        ) * (depth - kv_depth) + (d - kv_depth)
                        q_rope_ptr[dst_off] = (v_bf16 / qs).cast[rope_type]()

    # K_scale per token: max(|k_bf16|) / 448 across heads × kv_depth.
    for b in range(batch_size):
        for j in range(num_keys):
            var k_max = Float32(-1e10)
            for h in range(num_heads):
                for d in range(kv_depth):
                    var off = (
                        (b * num_keys + j) * num_heads + h
                    ) * kv_depth + d
                    var k_abs = abs(k_bf16_ptr[off]).cast[.float32]()
                    if k_abs > k_max:
                        k_max = k_abs
            k_scale_ptr[b * num_keys_padded + j] = max(
                k_max / Float32(448), Float32(1e-10)
            ).cast[scale_type]()

    # K (FP8) = k_bf16 / K_scale.
    for b in range(batch_size):
        for j in range(num_keys):
            var ks = k_scale_ptr[b * num_keys_padded + j].cast[.bfloat16]()
            for h in range(num_heads):
                for d in range(kv_depth):
                    var src_off = (
                        (b * num_keys + j) * num_heads + h
                    ) * kv_depth + d
                    var dst_off = (
                        (b * num_keys_padded + j) * num_heads + h
                    ) * kv_depth + d
                    k_ptr[dst_off] = (k_bf16_ptr[src_off] / ks).cast[qkv_type]()

    # V (FP8) = v_bf16 cast directly (no scaling on V in PTS).
    for b in range(batch_size):
        for j in range(num_keys):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var src_off = (
                        (b * num_keys + j) * num_heads + h
                    ) * kv_depth + d
                    var dst_off = (
                        (b * num_keys_padded + j) * num_heads + h
                    ) * kv_depth + d
                    v_ptr[dst_off] = v_bf16_ptr[src_off].cast[qkv_type]()

    # ------------------------------------------------------------------
    # Step 3: Build the row-offset tables. Strides use the PADDED
    # values so per-batch col_offsets are 4-aligned for the q_scale and
    # k_scale FP32 TMA loads.
    # ------------------------------------------------------------------
    var input_row_offsets_host = alloc[UInt32](batch_size + 1)
    var cache_row_offsets_host = alloc[UInt32](batch_size + 1)
    for i in range(batch_size):
        input_row_offsets_host[i] = UInt32(i * seq_len_padded)
        cache_row_offsets_host[i] = UInt32(i * num_keys_padded)
    input_row_offsets_host[batch_size] = UInt32(batch_size * seq_len_padded)
    cache_row_offsets_host[batch_size] = UInt32(batch_size * num_keys_padded)

    # ------------------------------------------------------------------
    # Step 4: Allocate paged K_rope blocks (BF16) + LUT + per-batch
    # cache lengths. Fill blocks with cache_bf16 / K_scale[per-token]
    # (matching the contiguous PTS test's quantization model where
    # ``cache stays in bf16, but it gets divided by k_scale to convert
    # into fp8 domain``). The kernel applies K_scale to the rope path
    # on read; the reference path uses the ORIGINAL cache_bf16 (before
    # division) so the kernel ↔ reference round-trip cancels the
    # scale.
    # ------------------------------------------------------------------
    # Page count uses `num_keys_padded` so that the K_rope paged cache
    # covers all rows the kernel reads (since the kernel's
    # `pos.num_keys = num_keys_padded` when `cache_lengths[i]=0` and
    # `input_row_offsets` stride is `seq_len_padded == num_keys_padded`
    # in this test). For our configs `ceildiv(17, 16) ==
    # ceildiv(20, 16) == 2`, so num_pages_per_batch is unchanged.
    var num_pages_per_batch = ceildiv(num_keys_padded, page_size)
    var total_pages = batch_size * num_pages_per_batch
    var max_pages_per_batch = lut_max_pages_per_batch(
        num_keys_padded, page_size
    )
    var lut_size = batch_size * max_pages_per_batch
    var block_elems = paged_block_elems(total_pages, page_size, CACHE_DEPTH)

    var blocks_host = alloc[Scalar[rope_type]](block_elems)
    var cache_lengths_host = alloc[UInt32](batch_size)
    var lookup_table_host = alloc[UInt32](lut_size)

    # Zero-init paged blocks. Tail slots past num_keys per batch stay
    # zero — the kernel may issue OOB sub-tile TMAs on partial-page
    # configs but those reads land in masked-out columns.
    for i in range(block_elems):
        blocks_host[i] = 0

    # For the SM100 paged path, ``cache_length(b)`` is interpreted as
    # the PRE-EXISTING cache length (start_pos in the kernel). Fresh
    # prefill ⇒ 0; the kernel attends to keys ``[0, seq_len)`` which
    # is where this test places its K_rope data.
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(0)

    fill_uniform_lookup_table(
        lookup_table_host,
        batch_size,
        num_keys_padded,
        page_size,
        max_pages_per_batch,
    )

    # Fill paged blocks per token, dividing cache_bf16 by K_scale.
    var pstride = page_stride(page_size, CACHE_DEPTH)
    var tstride = token_stride(CACHE_DEPTH)
    for b in range(batch_size):
        for j in range(num_keys):
            var ks = k_scale_ptr[b * num_keys + j].cast[.bfloat16]()
            var page_in_batch = j // page_size
            var slot_in_page = j % page_size
            # Uniform LUT: each batch's pages are contiguous in the
            # global block array.
            var global_page = b * num_pages_per_batch + page_in_batch
            var row_base = global_page * pstride + slot_in_page * tstride
            for d in range(CACHE_DEPTH):
                var src = cache_bf16_ptr[
                    ((b * num_keys + j) * KV_NUM_HEADS + 0) * CACHE_DEPTH + d
                ]
                blocks_host[row_base + d] = (src / ks).cast[rope_type]()

    # ------------------------------------------------------------------
    # Step 5: Allocate device buffers and copy from host.
    # ------------------------------------------------------------------
    var q_nope_device_buf = ctx.enqueue_create_buffer[qkv_type](q_nope_size)
    var q_rope_device_buf = ctx.enqueue_create_buffer[rope_type](q_rope_size)
    var q_scale_device_buf = ctx.enqueue_create_buffer[scale_type](q_scale_size)
    var k_device_buf = ctx.enqueue_create_buffer[qkv_type](k_size)
    var k_scale_device_buf = ctx.enqueue_create_buffer[scale_type](k_scale_size)
    var v_device_buf = ctx.enqueue_create_buffer[qkv_type](v_size)
    var output_device_buf = ctx.enqueue_create_buffer[output_type](o_size)
    var input_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_ro_buf = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var blocks_device = ctx.enqueue_create_buffer[rope_type](block_elems)
    var cache_lengths_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_device = ctx.enqueue_create_buffer[.uint32](lut_size)

    ctx.enqueue_copy(q_nope_device_buf, q_nope_ptr)
    ctx.enqueue_copy(q_rope_device_buf, q_rope_ptr)
    ctx.enqueue_copy(q_scale_device_buf, q_scale_ptr)
    ctx.enqueue_copy(k_device_buf, k_ptr)
    ctx.enqueue_copy(k_scale_device_buf, k_scale_ptr)
    ctx.enqueue_copy(v_device_buf, v_ptr)
    ctx.enqueue_copy(input_ro_buf, input_row_offsets_host)
    ctx.enqueue_copy(cache_ro_buf, cache_row_offsets_host)
    ctx.enqueue_copy(blocks_device, blocks_host)
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 6: Build TileTensors for the kernel call.
    # ------------------------------------------------------------------
    # All Q/K/V/scale TileTensor shapes use the PADDED per-batch
    # strides (`seq_len_padded` for Q-side, `num_keys_padded` for
    # K-side) so the kernel's TMA col_offsets — which derive from
    # `input_row_offsets` (Q) and `cache_row_offsets` (K, V, K_scale) —
    # are 4-aligned. The valid data lives in the first `seq_len` /
    # `num_keys` rows per batch; trailing rows are zero (Q/K/V) or
    # 1.0 (scales) so they don't perturb attention.
    var q_nope_device = TileTensor(
        q_nope_device_buf,
        row_major(
            (
                batch_size * seq_len_padded,
                Idx[num_heads],
                Idx[kv_depth],
            )
        ),
    )
    var q_rope_device = TileTensor(
        q_rope_device_buf,
        row_major(
            (
                batch_size * seq_len_padded,
                Idx[num_heads],
                Idx[(depth - kv_depth)],
            )
        ),
    )
    var q_scale_device = TileTensor(
        q_scale_device_buf,
        row_major((batch_size * seq_len_padded, Idx[1])),
    )
    var k_device = TileTensor(
        k_device_buf,
        row_major(
            (
                batch_size * num_keys_padded,
                Idx[num_heads],
                Idx[kv_depth],
            )
        ),
    )
    var k_scale_device = TileTensor(
        k_scale_device_buf,
        row_major((batch_size * num_keys_padded, Idx[1])),
    )
    var v_device = TileTensor(
        v_device_buf,
        row_major(
            (
                batch_size * num_keys_padded,
                Idx[num_heads],
                Idx[kv_depth],
            )
        ),
    )
    var output_device = TileTensor(
        output_device_buf,
        row_major(
            (
                batch_size * seq_len_padded,
                Idx[num_heads],
                Idx[kv_depth],
            )
        ),
    )
    var input_ro_tt = TileTensor(input_ro_buf, row_major(batch_size + 1))
    var cache_ro_tt = TileTensor(cache_ro_buf, row_major(batch_size + 1))

    # ------------------------------------------------------------------
    # Step 7: Build the PagedKVCacheCollection for K_rope. UNLIKE the
    # blockscale paged test, we do NOT pass `scale_dtype_` /
    # `quantization_granularity_` — per-token-scale uses a contiguous
    # K_scale TileTensor, not integrated per-(token, block) scales in
    # the paged collection.
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

    var blocks_lt = LayoutTensor[rope_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[rope_type, kv_params, page_size](
        LayoutTensor[rope_type, Layout.row_major[6]()](
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
        UInt32(seq_len_padded),  # max_seq_length
        UInt32(num_keys_padded),  # max_cache_length
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # ------------------------------------------------------------------
    # Step 8: Launch the kernel via the NEW paged-K_rope per-token-
    # scale `flare_mla_prefill` overload.
    # ------------------------------------------------------------------
    print("  Launching MLA prefill kernel (paged BF16 K_rope, PTS)...")

    flare_mla_prefill[rank=3](
        output_device,
        q_nope_device,
        q_rope_device,
        q_scale_device,
        k_device,
        k_scale_device,
        v_device,
        kv_cache,
        CausalMask(),
        input_ro_tt,
        cache_ro_tt,
        scale,
        ctx,
        q_max_seq_len=seq_len_padded,
    )

    ctx.synchronize()
    print("  Kernel completed (no crash).")

    # ------------------------------------------------------------------
    # Step 9: Copy the kernel output back to host.
    # ------------------------------------------------------------------
    ctx.enqueue_copy(output_ptr, output_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 10: Build the contiguous reference (Q_ref, K_ref, V_ref) by
    # dequantizing back to BF16. Because the kernel reads `cache /
    # K_scale` and applies K_scale on read, the reference uses the
    # ORIGINAL cache_bf16 (NOT the divided form) as K_ref's rope tail
    # — same pattern as ``test_mla_prefill_per_token_scale_fp8.mojo``.
    # ------------------------------------------------------------------
    # Reference tensor sizes use PADDED extents to match the kernel's
    # `pos.num_keys = num_keys_padded` view. Padded rows are zero-
    # filled (already from alloc[BFloat16] then explicit zero below);
    # they contribute nothing to attention and produce zero output —
    # same as the kernel side.
    var q_ref_size = batch_size * seq_len_padded * num_heads * depth
    var k_ref_size = batch_size * num_keys_padded * num_heads * depth
    var v_ref_size = k_ref_size
    var output_ref_size = batch_size * seq_len_padded * num_heads * depth

    var q_ref_host = alloc[BFloat16](q_ref_size)
    var k_ref_host = alloc[BFloat16](k_ref_size)
    var v_ref_host = alloc[BFloat16](v_ref_size)
    var output_ref_host = alloc[Scalar[output_type]](output_ref_size)

    # Zero-init padded reference tensors so the un-filled rows past
    # the logical extents don't contain uninitialised data.
    for i in range(q_ref_size):
        q_ref_host[i] = BFloat16(0)
    for i in range(k_ref_size):
        k_ref_host[i] = BFloat16(0)
    for i in range(v_ref_size):
        v_ref_host[i] = BFloat16(0)

    # Q_ref = [q_nope_fp8 * q_scale | q_rope_bf16 * q_scale]
    # Source from PADDED-stride q_nope/q_rope/q_scale buffers; write to
    # PADDED-stride q_ref. Only logical rows [0, seq_len) get real
    # values; padded rows stay zero from the init above.
    for b in range(batch_size):
        for s in range(seq_len):
            var qs = q_scale_ptr[b * seq_len_padded + s].cast[.bfloat16]()
            for h in range(num_heads):
                for d in range(kv_depth):
                    var dst = (
                        (b * seq_len_padded + s) * num_heads + h
                    ) * depth + d
                    var qn = (
                        q_nope_ptr[
                            ((b * seq_len_padded + s) * num_heads + h)
                            * kv_depth
                            + d
                        ]
                    ).cast[.bfloat16]()
                    q_ref_host[dst] = qn * qs
                for d in range(depth - kv_depth):
                    var dst = (
                        (b * seq_len_padded + s) * num_heads + h
                    ) * depth + (kv_depth + d)
                    var qr = q_rope_ptr[
                        ((b * seq_len_padded + s) * num_heads + h)
                        * (depth - kv_depth)
                        + d
                    ].cast[.bfloat16]()
                    q_ref_host[dst] = qr * qs

    # K_ref_nope = k_fp8 * k_scale
    # K_ref_rope = cache_bf16 (ORIGINAL, before division by K_scale)
    # V_ref     = v_fp8 cast back to BF16 (no scaling)
    # Padded rows past logical num_keys stay zero (k/v zero-init above
    # propagates to k_ref/v_ref).
    for b in range(batch_size):
        for j in range(num_keys):
            var ks = k_scale_ptr[b * num_keys_padded + j].cast[.bfloat16]()
            for h in range(num_heads):
                for d in range(kv_depth):
                    var src_off = (
                        (b * num_keys_padded + j) * num_heads + h
                    ) * kv_depth + d
                    var dst = (
                        (b * num_keys_padded + j) * num_heads + h
                    ) * depth + d
                    k_ref_host[dst] = k_ptr[src_off].cast[.bfloat16]() * ks
                    v_ref_host[dst] = v_ptr[src_off].cast[.bfloat16]()
                for d in range(depth - kv_depth):
                    var dst = (
                        (b * num_keys_padded + j) * num_heads + h
                    ) * depth + (kv_depth + d)
                    # Rope tail of K_ref is the ORIGINAL cache_bf16
                    # from the last `(depth - kv_depth)` slots of each
                    # cache row. cache_bf16_ptr is sized for LOGICAL
                    # num_keys (host reference buffer only).
                    k_ref_host[dst] = cache_bf16_ptr[
                        ((b * num_keys + j) * KV_NUM_HEADS + 0) * CACHE_DEPTH
                        + (CACHE_DEPTH - (depth - kv_depth) + d)
                    ]
                    # V has no rope component.
                    v_ref_host[dst] = 0

    # ------------------------------------------------------------------
    # Step 11: Run the naive MHA reference on the contiguous K_ref/V_ref.
    # ------------------------------------------------------------------
    var q_ref_device_buf = ctx.enqueue_create_buffer[.bfloat16](q_ref_size)
    var k_ref_device_buf = ctx.enqueue_create_buffer[.bfloat16](k_ref_size)
    var v_ref_device_buf = ctx.enqueue_create_buffer[.bfloat16](v_ref_size)
    var output_ref_device_buf = ctx.enqueue_create_buffer[output_type](
        output_ref_size
    )
    ctx.enqueue_copy(q_ref_device_buf, q_ref_host)
    ctx.enqueue_copy(k_ref_device_buf, k_ref_host)
    ctx.enqueue_copy(v_ref_device_buf, v_ref_host)

    # Reference TileTensors use PADDED extents so mha_gpu_naive
    # iterates over the same row counts the kernel sees.
    var q_ref_4d_device = TileTensor(
        q_ref_device_buf,
        row_major(
            (
                batch_size,
                seq_len_padded,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var k_ref_device = TileTensor(
        k_ref_device_buf,
        row_major(
            (
                batch_size,
                num_keys_padded,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var v_ref_device = TileTensor(
        v_ref_device_buf,
        row_major(
            (
                batch_size,
                num_keys_padded,
                Idx[num_heads],
                Idx[depth],
            )
        ),
    )
    var output_ref_device = TileTensor(
        output_ref_device_buf,
        row_major(
            (
                batch_size,
                seq_len_padded,
                Idx[num_heads],
                Idx[depth],
            )
        ),
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
        q_ref_4d_device.to_layout_tensor(),
        k_ref_operand,
        v_ref_operand,
        CausalMask(),
        output_ref_device.to_layout_tensor(),
        null_valid_length,
        scale,
        batch_size,
        seq_len_padded,
        num_keys_padded,
        num_heads,
        depth,
        1,  # group
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_ref_host, output_ref_device_buf)
    ctx.synchronize()

    # ------------------------------------------------------------------
    # Step 12: Compare. FP8 quantization noise + bf16 accumulation
    # imprecision push the tolerance higher than the bf16-only generic
    # path. The contiguous PTS test (`test_mla_prefill_per_token_scale_fp8.mojo`)
    # uses 5e-2 with large num_keys (120/1180) where noise averages
    # out. Our small partial-page configs (num_keys as small as 17)
    # push noise much higher; we adopt the same tolerance the paged
    # FP8 decode test uses (`test_mla_decode_blockwise_fp8.mojo:481–482`).
    # ------------------------------------------------------------------
    comptime atol: Float64 = 3e-1
    comptime rtol: Float64 = 5e-2
    var max_abs_err = Float64(0)
    # Compare only the first `seq_len` Q rows per batch (the logical
    # extent). The padded Q rows produce zero output on both kernel
    # and reference sides — verifying them adds no signal.
    # output_ptr/output_ref_host both use seq_len_padded stride.
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(kv_depth):
                    var actual = output_ptr.load(
                        (b * seq_len_padded + s) * num_heads * kv_depth
                        + h * kv_depth
                        + d
                    ).cast[.float64]()
                    var expect = output_ref_host.load(
                        ((b * seq_len_padded + s) * num_heads + h) * depth + d
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
                    assert_almost_equal(actual, expect, atol=atol, rtol=rtol)

    print("  PASS, max_abs_err:", max_abs_err)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    q_nope_ptr.free()
    q_rope_ptr.free()
    q_scale_ptr.free()
    k_ptr.free()
    k_scale_ptr.free()
    v_ptr.free()
    output_ptr.free()
    q_bf16_ptr.free()
    k_bf16_ptr.free()
    v_bf16_ptr.free()
    cache_bf16_ptr.free()
    blocks_host.free()
    cache_lengths_host.free()
    lookup_table_host.free()
    input_row_offsets_host.free()
    cache_row_offsets_host.free()
    q_ref_host.free()
    k_ref_host.free()
    v_ref_host.free()
    output_ref_host.free()

    _ = q_nope_device_buf
    _ = q_rope_device_buf
    _ = q_scale_device_buf
    _ = k_device_buf
    _ = k_scale_device_buf
    _ = v_device_buf
    _ = output_device_buf
    _ = input_ro_buf
    _ = cache_ro_buf
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_ref_device_buf
    _ = k_ref_device_buf
    _ = v_ref_device_buf
    _ = output_ref_device_buf


def main() raises:
    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Iterate over every ``num_keys`` in the shared list at
            # runtime; re-seed per ``num_keys`` for independent
            # reproducibility regardless of iteration order.
            for num_keys in num_keys_to_test():
                seed(0)
                # Single-batch baseline.
                run_test_paged_prefill_per_token_scale[
                    qkv_type=DType.float8_e4m3fn,
                    rope_type=DType.bfloat16,
                    scale_type=DType.float32,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    kv_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                ](num_keys, num_keys, ctx)
                # Multi-batch — exercises cross-batch LUT layout: each
                # batch's pages occupy a distinct slice of the global
                # block array, so any incorrect LUT lookup past a
                # batch's valid page count would point to another
                # batch's data.
                run_test_paged_prefill_per_token_scale[
                    qkv_type=DType.float8_e4m3fn,
                    rope_type=DType.bfloat16,
                    scale_type=DType.float32,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    kv_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=2,
                ](num_keys, num_keys, ctx)
