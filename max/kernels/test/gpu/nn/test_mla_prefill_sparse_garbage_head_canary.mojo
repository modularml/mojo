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
"""Cross-row leak in the SM100 sparse MLA prefill kernel's O-rescale gate.

`mla_prefill_sparse`'s per-k-block online-softmax decided whether to adopt
a grown running max (`new_max`/`mi`/`li`) with a warp-wide ANY-ballot
instead of each lane's own predicate. `idx_in_wg` (0..127, `WARPGROUP_SIZE`)
gives one INDEPENDENT attention head-row per lane -- threads `t` and `t^64`
are the SAME head's two K-column halves (redundant, combined via
`rowwise_max_ptr`), but lanes `0..31` (one physical warp) are 32 DISTINCT
heads. A sibling head whose growth crossed the threshold forced every
other head in the warp to adopt its grown max too, substituting a real
(non-1.0) `scale_for_old` for a head whose own growth never asked for it.

The fix is NOT simply "make `should_scale_o` per-lane and drop the vote":
the O-rescale WALK gated by that decision touches TMEM via
`tcgen05_ld`/`tcgen05_st` with `datapaths=32`, which are warp-collective
hardware ops requiring every lane to participate uniformly. A first attempt
at a pure per-lane predicate made those lanes diverge on the collective op
and hung the kernel (three separate hangs during development, each killed
at its bazel sandbox timeout, before this was diagnosed). The landed fix
keeps a warp-uniform ANY-vote to gate WHETHER the walk runs at all, but
computes the actual state (`new_max`/`mi`/`li`/`scale_for_old`'s VALUE)
strictly per-lane first; a lane coerced into the walk that didn't need a
rescale applies its own `scale_for_old = 1.0`, an exact no-op multiply --
the same "coerced-but-harmless" shape the audit found BENIGN elsewhere
(`softmax_warp.mojo`'s BLASST skip vote, the correction-warp sibling
ballots).

At `num_heads=32` every real head fits in warp 0 (`idx_in_wg` 0..31 exactly),
so this canary holds head 0's Q byte-identical across two runs and poisons
every OTHER head's Q with large-magnitude garbage in one of them (K/V,
indices, and page mapping stay identical throughout -- see the re-seeding
note below); head 0's output must not move.

Shape (`num_kv_tokens=1024, topk=256` = 4 k-blocks, `seq_len=32`, matching
the existing `b1_s32_h32_kv1024_topk256` correctness case) is deliberately
an ALREADY-COVERED production shape, not a bespoke minimal one: a
from-scratch single-page reconstruction of this kernel's paged-cache
plumbing, and later that same multi-page shape at `seq_len=1`, both made
the kernel spin indefinitely under `bazel test` (independently of the
should_scale_o fix above -- these hangs reproduced even against the
unmodified kernel, so they are this test's OWN construction talking past
some grid/pipeline assumption, not a kernel defect), so this canary reuses
`test_mla_prefill_sparse.mojo`'s exact KV/indices/page-shuffle construction
verbatim (via `_run_prefill_sparse_diff`, a copy of that file's
`run_test_prefill_sparse` with the host-reference/tolerance check removed
and a `q_override` injection point added) instead of re-deriving it.

`_run_prefill_sparse_diff` has no `kv_override` hook -- it generates KV
internally via `randn`, exactly like the file it was copied from. The
caller MUST re-seed with the identical `rng_seed` immediately before EACH
of the two calls (see `test_garbage_head_canary`), or the two runs' internal
KV draws diverge and EVERY element differs regardless of the kernel's
correctness -- a methodology bug this canary hit once during development
(it produced a deceptively strong "100% of elements differ, every seed"
signal that had nothing to do with the warp-vote defect) before the
re-seed fix.

The identical `should_scale_o` pattern (same line shape, verified by direct
diff) appears in the two native-FP8 sibling kernels
(`mla_prefill_sparse_qkv_fp8.mojo`, `mla_prefill_sparse_kv_fp8.mojo`), fixed
in the same commit; this file exercises the shared code path (the BF16-KV
kernel) that all three forked from.

This kernel is prefill-only (there is no sparse-MLA decode path sharing this
file), so there is no draft-vs-committed-token asymmetry to exploit --
the poison is a same-launch sibling HEAD with different Q, not a
never-committed draft token.

CONFIRMED on SM100 (B200) with the corrected re-seeding methodology:
pre-fix, 8/24 seeds fail with a real, data-dependent PARTIAL mismatch (1 to
210 of head 0's 512 output elements, not "every element" -- consistent with
the leaked `scale_for_old` being a genuine but often-small perturbation);
post-fix, 0/24 seeds fail, every element bit-exact.
"""

from std.math import ceildiv, sqrt
from std.memory import UnsafePointer, alloc
from std.random import randn, randn_float64, seed
from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_equal, assert_true

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
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse import mla_prefill_sparse
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# Test constants (matches test_mla_prefill_sparse.mojo)
# ===-----------------------------------------------------------------------===#

comptime KV_LORA_RANK = 512
comptime QK_ROPE_HEAD_DIM = 64
comptime QK_DEPTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
comptime V_DEPTH = KV_LORA_RANK  # 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1
comptime SOFTMAX_SCALE_BASE_DIM = 192


# ===-----------------------------------------------------------------------===#
# Helpers (matches test_mla_prefill_sparse.mojo)
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
# Setup + kernel launch, copied from test_mla_prefill_sparse.mojo's
# `run_test_prefill_sparse` with the host-reference/tolerance check removed
# and a `q_override` injection point added (see module docstring for why).
# ===-----------------------------------------------------------------------===#


def _run_prefill_sparse_diff[
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
    q_override: List[Float64] = [],
) raises -> List[Float64]:
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
    when `top_k_length == topk`).

    `q_override` (optional): a flat `[q_elems]` list of values to write
    into Q instead of the internal `randn` fill, letting a caller build
    a specific per-head/per-row pattern (e.g. poison one head's Q while
    holding another's fixed) without duplicating this setup. Returns the
    raw kernel output (no reference computed, no tolerance check) for
    the caller to compare directly.
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
    if len(q_override) > 0:
        for i in range(q_elems):
            q_host[i] = Scalar[q_type](q_override[i])

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
    # (No host reference here -- this helper returns raw output for a
    # differential caller comparison, not a numeric-correctness check.)
    # -----------------------------------------------------------------------
    var out_elems = total_q_tokens * num_heads * V_DEPTH
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
    # Read output back to host
    # -----------------------------------------------------------------------
    var out_host = alloc[Scalar[q_type]](out_elems)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var result = List[Float64](length=out_elems, fill=Float64(0))
    for i in range(out_elems):
        result[i] = out_host[i].cast[.float64]()

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
    out_host.free()
    h_indices.free()
    h_topk_lengths.free()
    return result^


# ===-----------------------------------------------------------------------===#
# Differential canary
# ===-----------------------------------------------------------------------===#


def test_garbage_head_canary(ctx: DeviceContext, rng_seed: Int) raises:
    """Poison every head EXCEPT head 0 (all share warp 0); head 0's
    output must not move."""
    print("  [DSA sparse-prefill garbage-head canary] seed=", rng_seed, sep="")
    seed(rng_seed)

    comptime num_heads = 32
    comptime topk = 256
    comptime seq_len = 32  # matches test_mla_prefill_sparse.mojo's
    # "b1_s32_h32_kv1024_topk256" shape exactly -- seq_len=1 made the kernel
    # spin indefinitely (see module docstring); this canary poisons only
    # query row 0's heads[1:32] and leaves rows 1..31 as ordinary matching
    # random data in both runs.
    var q_elems = seq_len * num_heads * QK_DEPTH

    var q_clean = List[Float64](length=q_elems, fill=Float64(0))
    for i in range(q_elems):
        q_clean[i] = Float64(randn_float64(0.0, 8.0))

    var q_poison = List[Float64](length=q_elems, fill=Float64(0))
    for i in range(q_elems):
        q_poison[i] = q_clean[i]
    for h in range(1, num_heads):  # row 0's heads[1:32] only.
        for d in range(QK_DEPTH):
            q_poison[h * QK_DEPTH + d] = Float64(30.0)

    # `_run_prefill_sparse_diff` has no kv_override hook -- it generates
    # KV internally via `randn`. Re-seed immediately before EACH call so
    # both invocations replay the identical internal random sequence
    # (same KV, same page shuffle) and only the externally-injected Q
    # differs between them.
    seed(rng_seed)
    var o_clean = _run_prefill_sparse_diff[.bfloat16, num_heads, topk](
        "garbage_head_clean",
        1,
        seq_len,
        1024,
        ctx,
        q_override=q_clean,
    )
    seed(rng_seed)
    var o_poison = _run_prefill_sparse_diff[.bfloat16, num_heads, topk](
        "garbage_head_poison",
        1,
        seq_len,
        1024,
        ctx,
        q_override=q_poison,
    )

    var mismatches = 0
    for i in range(V_DEPTH):
        if o_clean[i] != o_poison[i]:
            mismatches += 1
    print("     head0 mismatches=", mismatches, "/", V_DEPTH, sep="")
    assert_equal(
        mismatches,
        0,
        (
            "head 0 changed when sibling heads in its warp were"
            " poisoned -- cross-row leak in the kernel"
        ),
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            var failures = List[String]()
            for s in range(24):
                try:
                    test_garbage_head_canary(ctx, s)
                except e:
                    failures.append(String("seed=") + String(s))
                    print("     CANARY FAILED (seed=", s, "): ", e, sep="")

            assert_true(
                len(failures) == 0,
                "DSA sparse-prefill garbage-head canary failed for: "
                + String(failures),
            )
        else:
            pass
