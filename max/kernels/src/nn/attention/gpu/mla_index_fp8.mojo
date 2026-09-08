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
"""MLA FP8 index kernel for computing attention scores with paged KV cache."""

from std.sys import get_defined_int, size_of
from std.sys.info import _has_blackwell_tcgen05
from std.math import align_up, ceildiv, clamp

from layout import (
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, FuncAttribute

from kv_cache.types import KVCollectionT

from nn.index_fp8 import fp8_index_kernel, IndexSmemStorage
from nn.attention.gpu.sparse_index_fp8_sm100 import (
    _BM_KEY,
    SPEC_DECODE_N_TOKENS_ALT,
    fp8_index_score_sm100,
)
from nn.attention.mha_mask import MHAMask, MaskName
from nn.attention.mha_operand import KVCacheMHAOperand, KVCacheScalesMHAOperand
from nn.attention.mha_utils import dispatch_mask, indexer_key_bound
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    persistent_topk_block_split,
)
from nn.topk import topk_gpu

from std.utils.index import Index


# Peak bytes of the transient score matrix. Matches vLLM's
# `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` default (512) so a comparison against
# their indexer measures the kernels rather than two different memory policies.
comptime _SCORES_BUDGET_BYTES = (
    get_defined_int["MLA_INDEX_SCORES_BUDGET_MB", 512]() * 1024 * 1024
)


# ===----------------------------------------------------------------------=== #
# Mask application kernel
# ===----------------------------------------------------------------------=== #


@__name(t"mla_apply_mask")
def apply_mask_kernel[
    mask_t: MHAMask,
    ScoresLayoutType: TensorLayout,
    scores_origin: MutOrigin,
    VLLayoutType: TensorLayout,
    vl_origin: ImmOrigin,
    CLLayoutType: TensorLayout,
](
    output: TileTensor[.float32, ScoresLayoutType, scores_origin],
    valid_length: TileTensor[.uint32, VLLayoutType, vl_origin],
    cache_lengths: TileTensor[.uint32, CLLayoutType, ImmutAnyOrigin],
    mask: mask_t,
    max_num_keys: Int32,
):
    """Apply causal mask to the output scores.

    Parameters:
        mask_t: The `MHAMask` type applied to each score coordinate.
        ScoresLayoutType: Layout of the `output` scores tensor.
        scores_origin: Origin of the `output` scores tensor.
        VLLayoutType: Layout of the `valid_length` tensor.
        vl_origin: Origin of the `valid_length` tensor.
        CLLayoutType: Layout of the `cache_lengths` tensor.

    Args:
        output: Score matrix with row stride `max_num_keys`, indexed as
            `[global_seq_idx, key_idx]`.
        valid_length: Row offsets into `output` per batch, length
            `batch_size + 1`.
        cache_lengths: Per-batch cached-prefix length used to map a local
            query index to an absolute position.
        mask: The mask instance applied to each score coordinate.
        max_num_keys: Row stride of `output` and maximum keys per token.
    """
    var batch_idx = block_idx.x
    var seq_idx = block_idx.y * 16 + thread_idx.x
    var key_idx = block_idx.z * 16 + thread_idx.y

    var start_of_seq = valid_length.raw_load(batch_idx)
    var end_of_seq = valid_length.raw_load(batch_idx + 1)
    var seq_len = end_of_seq - start_of_seq

    if seq_idx >= Int(seq_len) or key_idx >= Int(max_num_keys):
        return

    var global_seq_idx = start_of_seq + UInt32(seq_idx)
    var current_val = output.raw_load(
        Int(global_seq_idx) * Int(max_num_keys) + key_idx
    )

    # Apply mask: coord = [batch, head, query_idx, key_idx]
    # key_idx indexes the full KV cache, so the query index must be its
    # absolute position cache_len + seq_idx.
    var cache_len = Int(cache_lengths[batch_idx])
    var coord = Index(batch_idx, 0, cache_len + seq_idx, key_idx)
    var masked_val = mask.mask(coord, current_val)

    output.raw_store(
        Int(global_seq_idx) * Int(max_num_keys) + key_idx, masked_val
    )


@__name(t"mla_fill_invalid_topk_{use_causal_mask}_{kpool}")
def fill_invalid_topk_kernel[
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    cache_lengths_layout: TensorLayout,
    use_causal_mask: Bool,
    kpool: Int = 1,
](
    output_indices: UnsafePointer[Int32, MutAnyOrigin],
    topk_indices: UnsafePointer[Int32, MutAnyOrigin],
    input_row_offsets: TileTensor[.uint32, IROLayoutType, iro_origin],
    cache_lengths: TileTensor[.uint32, cache_lengths_layout, ImmutAnyOrigin],
    total_seq_len: Int32,
    top_k: Int32,
    effective_k: Int32,
):
    """Scatter the compact top-k indices into the top_k-strided output, with
    invalid positions set to -1.

    topk_gpu wrote valid indices to positions [0, effective_k) of the compact
    `topk_indices` buffer (row stride effective_k). This kernel copies them
    into `output_indices` (row stride top_k), setting positions that should be
    -1:
    - Positions [effective_k, top_k): no computed value.
    - Positions where k_idx >= num_keys for that token.
    - Positions where the index VALUE >= num_keys (topk selected an invalid key).

    Output shape: [total_seq_len, top_k].

    With causal masking, each token can only see keys up to its position:
        num_keys = cache_len + local_seq_idx + 1
    Without causal masking, each token can see all keys in the batch:
        num_keys = cache_len + seq_len

    Parameters:
        IROLayoutType: Layout of the `input_row_offsets` tensor.
        iro_origin: Origin of the `input_row_offsets` tensor.
        cache_lengths_layout: Layout of the `cache_lengths` tensor.
        use_causal_mask: Whether each token is restricted to keys up to
            its own position.
        kpool: Tokens per pooled cache row. `1` scores one row per token;
            `k > 1` scores one pooled key per `k` consecutive tokens, so
            every candidate count and the caller's `top_k` are
            pool-granular.

    Args:
        output_indices: Output buffer of shape `[total_seq_len, top_k]`
            with invalid positions set to -1.
        topk_indices: Compact top-k index buffer of shape
            `[total_seq_len, effective_k]` produced by `topk_gpu`.
        input_row_offsets: Ragged row offsets per batch, length
            `batch_size + 1`.
        cache_lengths: Per-batch cached-prefix length used to compute the
            number of keys each token may attend to.
        total_seq_len: Number of token rows in `output_indices`.
        top_k: Row stride of `output_indices` and the requested number of
            selections per token.
        effective_k: Row stride of `topk_indices` and the actual number of
            computed selections, `min(top_k, max_num_keys)`.
    """
    comptime assert cache_lengths.flat_rank == 1

    var _total_seq_len = Int(total_seq_len)
    var _top_k = Int(top_k)
    var _effective_k = Int(effective_k)
    var token_idx = block_idx.x

    if token_idx >= _total_seq_len:
        return

    # Token-level quantities (independent of k): find which batch this token
    # belongs to and how many keys it may attend to.
    var batch_idx = 0
    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    for b in range(batch_size):
        var q_end_b = Int(input_row_offsets.raw_load(b + 1))
        if token_idx < q_end_b:
            batch_idx = b
            break

    var q_start = Int(input_row_offsets.raw_load(batch_idx))
    var q_end = Int(input_row_offsets.raw_load(batch_idx + 1))
    var seq_len = q_end - q_start
    var local_seq_idx = token_idx - q_start

    var cache_len = Int(cache_lengths[batch_idx])

    # Compute num_keys based on mask type
    var num_keys = indexer_key_bound[kpool](
        cache_len + seq_len, seq_len, local_seq_idx, Int(use_causal_mask)
    )

    # Cover ALL _top_k output columns. The launch caps block_dim at 1024, which
    # is smaller than _top_k when _top_k > 1024 (e.g. the indexer's _top_k=2048),
    # so each thread strides across multiple columns. Without this grid-stride
    # loop, columns [block_dim, _top_k) were never written (garbage, not -1).
    var k_idx = Int(thread_idx.x)
    while k_idx < _top_k:
        # Output index: [token_idx, k_idx] with _top_k stride
        var out_idx = Int(token_idx) * _top_k + k_idx

        if k_idx >= _effective_k:
            # No computed value at this position: must be -1.
            output_indices[out_idx] = -1
        else:
            # Read topk's selection from the compact (_effective_k-strided)
            # buffer, then invalidate it if it is out of range:
            # 1. position beyond the valid keys (k_idx >= num_keys)
            # 2. the index VALUE points beyond valid keys (idx_val >= num_keys),
            #    which can happen because topk operates on
            #    max_num_keys >= num_keys for this token/batch.
            var idx_val = Int(
                topk_indices[Int(token_idx) * _effective_k + k_idx]
            )
            if k_idx >= num_keys or idx_val >= num_keys or idx_val < 0:
                output_indices[out_idx] = -1
            else:
                output_indices[out_idx] = Int32(idx_val)

        k_idx += Int(block_dim.x)


@__name(t"mla_topk_row_bounds_{use_causal_mask}_{kpool}")
def topk_row_bounds_kernel[
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    cache_lengths_layout: TensorLayout,
    use_causal_mask: Bool,
    kpool: Int = 1,
](
    row_bounds: UnsafePointer[Int32, MutAnyOrigin],
    input_row_offsets: TileTensor[.uint32, IROLayoutType, iro_origin],
    cache_lengths: TileTensor[.uint32, cache_lengths_layout, ImmutAnyOrigin],
    total_seq_len: Int32,
    max_num_keys: Int32,
):
    """Compute each token row's live-key count for the bounded top-k.

    Writes `row_bounds[token] = min(num_keys, max_num_keys)` with `num_keys`
    from the shared `indexer_key_bound` helper (causal:
    `cache_len + local_seq_idx + 1`; non-causal: `cache_len + seq_len`).

    This is exactly the range the scorers write for that row (they compute
    the same helper's bound), so a top-k clamped to it reads only written
    score slots. `max_num_keys` may be a capture-time upper bound far above
    the batch's real lengths; the clamp keeps every bound within the row
    stride.

    Parameters:
        IROLayoutType: Layout of the `input_row_offsets` tensor.
        iro_origin: Origin of the `input_row_offsets` tensor.
        cache_lengths_layout: Layout of the `cache_lengths` tensor.
        use_causal_mask: Whether each token is restricted to keys up to its
            own position.
        kpool: Tokens per pooled cache row. `1` scores one row per token;
            `k > 1` scores one pooled key per `k` consecutive tokens, so
            every candidate count and the caller's `top_k` are
            pool-granular.

    Args:
        row_bounds: Output buffer of shape `[total_seq_len]`.
        input_row_offsets: Ragged row offsets per batch, length
            `batch_size + 1`.
        cache_lengths: Per-batch cached-prefix length.
        total_seq_len: Number of token rows.
        max_num_keys: Row stride of the scores buffer (upper bound on any
            row's key count).
    """
    comptime assert cache_lengths.flat_rank == 1

    var token_idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if token_idx >= Int(total_seq_len):
        return

    var batch_idx = 0
    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    for b in range(batch_size):
        var q_end_b = Int(input_row_offsets.raw_load(b + 1))
        if token_idx < q_end_b:
            batch_idx = b
            break

    var q_start = Int(input_row_offsets.raw_load(batch_idx))
    var q_end = Int(input_row_offsets.raw_load(batch_idx + 1))
    var seq_len = q_end - q_start
    var local_seq_idx = token_idx - q_start

    var cache_len = Int(cache_lengths[batch_idx])
    var num_keys = indexer_key_bound[kpool](
        cache_len + seq_len, seq_len, local_seq_idx, Int(use_causal_mask)
    )
    row_bounds[token_idx] = Int32(min(num_keys, Int(max_num_keys)))


# ===----------------------------------------------------------------------=== #
# Main function: mla_indexer_ragged_float8_paged
# ===----------------------------------------------------------------------=== #


@always_inline
def mla_indexer_ragged_float8_paged[
    dtype: DType,
    KCollectionT: KVCollectionT,
    num_heads: Int,
    depth: Int,
    top_k: Int,
    mask_str: StaticString,
    scores_budget_bytes: Int = _SCORES_BUDGET_BYTES,
    kpool: Int = 1,
](
    output_indices: TileTensor[.int32, ...],
    q: TileTensor[mut=False, dtype, ...],
    q_s: TileTensor[.float32, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    k_collection: KCollectionT,
    layer_idx: UInt32,
    ctx: DeviceContext,
) raises:
    """Compute FP8 indexed attention scores using paged KV cache and return top-k indices.

    This function:
    1. Computes FP8 matmul between q and cached k (with scales), aggregated across heads
    2. Applies the specified mask (causal, etc.)
    3. Computes top-k indices per token (scores are summed across all heads)

    Parameters:
        dtype: Element type of the `q` query tensor, an FP8 dtype.
        KCollectionT: Type of the KV collection holding cached K values and
            K scales.
        num_heads: Number of attention heads per token.
        depth: Per-head key dimension (head size) in elements.
        top_k: Requested number of top-scoring key indices to select per
            token.
        mask_str: Name of the mask to apply, either `MaskName.NULL` or
            `MaskName.CAUSAL`.
        scores_budget_bytes: Peak bytes the transient score matrix may occupy.
            Longer batches are scored a row-window at a time to stay under it
            (see the chunking below). Exposed so tests can force a window small
            enough to exercise the multi-chunk path on toy shapes.
        kpool: Tokens per pooled cache row. `1` scores one row per token;
            `k > 1` scores one pooled key per `k` consecutive tokens, so
            every candidate count and the caller's `top_k` are
            pool-granular.

    Args:
        output_indices: Dense output tensor for top-k indices [total_seq_len, top_k].
            Invalid positions (where there are fewer than top_k valid keys due to
            causal masking or shorter sequences) are filled with -1.
        q: Query tensor [total_seq_len, num_heads, head_dim] in FP8.
        q_s: Query scales [total_seq_len, num_heads] in float32.
        input_row_offsets: Ragged row offsets for queries [batch_size + 1].
        k_collection: KV collection containing cached K values and K scales.
            K scales are accessed via k_cache.scales (quantization_granularity=head_size).
        layer_idx: Layer index for retrieving cache.
        ctx: Device context.
    """
    # Verify that k_collection has scales enabled (required for MLA k_s).
    # For MLA, scales should have head_dim_granularity == 1 (one scale per token
    # per head), which requires quantization_granularity >= depth (head_size).
    comptime CacheType = KCollectionT.CacheType
    comptime assert (
        CacheType.quantization_enabled
    ), "k_collection must have quantization/scales enabled for MLA k_s values"
    comptime assert CacheType.quantization_granularity >= depth, (
        "k_collection.quantization_granularity must be >= depth (head_dim) for"
        " MLA (requires one scale per token per head, i.e. head_dim_granularity"
        " == 1)"
    )

    # Only NULL (no mask) and CAUSAL masks are supported
    comptime assert (
        mask_str == MaskName.NULL.name or mask_str == MaskName.CAUSAL.name
    ), "mask_str must be either MaskName.NULL or MaskName.CAUSAL"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    var total_seq_len = Int(q.dim[0]())
    if total_seq_len == 0:
        return

    var k_cache = k_collection.get_key_cache(Int(layer_idx))

    # max_new_tokens is used for grid dimensions (maximum possible new tokens)
    var max_new_tokens = Int(k_cache.max_prompt_length())

    # An upper bound on the candidate rows per token. Under graph-capture
    # replay this holds the capture-time bound, far above the batch's real
    # lengths, so use it only to size allocations and grid dims, never to size
    # per-step work. With `kpool > 1` a cache row is a pooled key covering
    # `kpool` tokens, so counts from here down are pool-granular, `top_k`
    # included.
    var max_num_keys = (
        Int(k_cache.max_context_length()) + max_new_tokens
    ) // kpool

    var effective_k = min(top_k, max_num_keys)

    var k_operand = KVCacheMHAOperand(k_cache)
    var ks_operand = KVCacheScalesMHAOperand(k_cache)

    comptime use_sm100_scorer = (
        _has_blackwell_tcgen05()
        and num_heads in (64, 32, 8, 4)
        and depth == 128
        and (
            type_of(k_operand).page_size == 0
            or type_of(k_operand).page_size % _BM_KEY == 0
        )
    )

    comptime assert kpool == 1 or use_sm100_scorer, (
        "pooled indexing (kpool > 1) is implemented only on the SM100"
        " tensor-core scorer; the scalar fallback walks token rows"
    )

    # Per-batch KV cache lengths (cached-prefix length). Needed by the row-bound
    # kernel below, by the causal mask pass on the scalar path (to map a local
    # query index to an absolute position), and by fill_invalid_topk.
    var cache_lengths = k_cache.cache_lengths_nd()

    comptime use_causal_mask = mask_str != MaskName.NULL.name

    # The score matrix is `total_seq_len x max_num_keys` f32 -- at long context
    # the largest allocation this op makes by an order of magnitude (~1.2 GB at
    # 4096 tokens over 76K keys) and what caps `--max-batch-input-tokens`. Score
    # it one ROW WINDOW at a time into one budget-sized buffer instead: a
    # chunk's rows are consumed by the top-k before the next overwrites them, so
    # peak scratch is `rows_per_chunk x max_num_keys` however large the batch
    # grows.
    #
    # Looping on the host is safe because prefill is never under device graph
    # capture (built at a fixed decode query width; replay rejects any other).
    # The window is over ROWS, not requests: `input_row_offsets` is device data
    # the host cannot read without a sync, while row boundaries are host-known
    # and every stage after the scorer is row-parallel over a base pointer and a
    # count, so they slice by pointer arithmetic alone.
    #
    # Only the SM100 scorers take a window, so the scalar fallback (and its
    # separate mask pass) stays one whole-batch chunk, and `topk_gpu` below is
    # unreachable while chunking: `top_k <= PERSISTENT_TOPK_MAX_N` bounds
    # `effective_k` under its threshold.
    comptime chunk_scores = use_sm100_scorer and top_k <= PERSISTENT_TOPK_MAX_N
    var rows_per_chunk = total_seq_len
    comptime if chunk_scores:
        var row_bytes = max_num_keys * size_of[DType.float32]()
        # Largest window the budget allows, remainder to a short final chunk.
        # Spreading the rows evenly instead -- same chunk count, no runt --
        # reads better and measured 8% SLOWER at 2048 and 4096 tokens: a chunk's
        # grid is `batch x ceildiv(min(max_seq_len, chunk_rows), N_TOKENS)`, so
        # a short final chunk launches a proportionally short grid and costs
        # almost nothing, while an even split turns that nearly-free tail into a
        # full-price one.
        rows_per_chunk = clamp(
            scores_budget_bytes // row_bytes, 1, total_seq_len
        )

    var scores_buf = ctx.enqueue_create_buffer[.float32](
        rows_per_chunk * max_num_keys
    )

    # Per-row live-key bounds let the top-k scan each row's real length instead
    # of the full max_num_keys stride. They depend only on the ragged metadata,
    # so this runs once for the whole batch and each chunk reads its own slice.
    var row_bounds_buf = ctx.enqueue_create_buffer[.int32](total_seq_len)
    var row_bounds_ptr = rebind[UnsafePointer[Int32, MutAnyOrigin]](
        row_bounds_buf.unsafe_ptr()
    )
    if effective_k <= PERSISTENT_TOPK_MAX_N:
        comptime bounds_kernel = topk_row_bounds_kernel[
            input_row_offsets.LayoutType,
            ImmOrigin(input_row_offsets.origin),
            type_of(cache_lengths).LayoutType,
            use_causal_mask,
            kpool,
        ]
        ctx.enqueue_function[bounds_kernel](
            row_bounds_ptr,
            input_row_offsets.as_immut(),
            cache_lengths,
            Int32(total_seq_len),
            Int32(max_num_keys),
            grid_dim=(ceildiv(total_seq_len, 128), 1, 1),
            block_dim=(128, 1, 1),
        )

    # topk_gpu strides its index output by effective_k, so when effective_k <
    # top_k we use a compact buffer here and scatter into the top_k-strided
    # output_indices in fill_invalid (writing directly would misplace rows).
    var topk_vals_buf = ctx.enqueue_create_buffer[.float32](
        total_seq_len * effective_k
    )
    var topk_vals_tile = TileTensor(
        topk_vals_buf,
        row_major(total_seq_len, effective_k),
    )

    var topk_idxs_buf = ctx.enqueue_create_buffer[.int32](
        total_seq_len * effective_k
    )
    var topk_idxs_tile = TileTensor(
        topk_idxs_buf,
        row_major(total_seq_len, effective_k),
    )

    for chunk_begin in range(0, total_seq_len, rows_per_chunk):
        var chunk_rows = min(rows_per_chunk, total_seq_len - chunk_begin)
        var scores_tile = TileTensor(
            scores_buf,
            row_major(chunk_rows, max_num_keys),
        )

        # -inf-fill only where a consumer reads past a row's live range: the
        # scalar scorer's mask pass and the topk_gpu fallback. The SM100 scorers
        # write every live slot and the bounded top-k reads only those, so there
        # the fill would be max_num_keys-proportional waste.
        if not use_sm100_scorer or effective_k > PERSISTENT_TOPK_MAX_N:
            scores_buf.enqueue_fill(-Float32.MAX)

        comptime if use_sm100_scorer:
            fp8_index_score_sm100[
                dtype,
                type_of(k_operand),
                type_of(ks_operand),
                num_heads,
                depth,
                _is_cache_length_accurate=False,
                # Speculative-decode tile: 3 divides a 6-token MTP step, which
                # the default 4-token tile at nh=32 covers only by spending 256
                # MMA columns on 192 live ones. Inert at every other head count.
                #
                # The `max_seq_len` this entry passes is `max_prompt_length()`,
                # the batch maximum of NEW tokens -- not a context length -- so
                # the reachability bound is "no request in the batch brings more
                # than 9 new tokens", not "not a prefill". A short prompt, a
                # chunked-prefill final chunk, or a prefix-cache-hit tail of 3,
                # 6 or 9 tokens does reach this tile when some entry's cache
                # makes `max_num_keys` deep enough to open the key-split arm.
                # That is intended: at <= 9 query tokens against >= 8065 keys
                # the launch is decode-shaped by every measure the route uses,
                # and it is already on the key-split arm without this hint --
                # the tile only makes its MMA columns exact. What cannot happen
                # is a many-token prefill landing here.
                N_TOKENS_ALT=SPEC_DECODE_N_TOKENS_ALT,
                kpool=kpool,
            ](
                scores_tile,
                q,
                q_s.as_immut(),
                k_operand,
                ks_operand,
                input_row_offsets,
                batch_size,
                max_new_tokens,
                max_num_keys,
                mask_str == MaskName.CAUSAL.name,
                ctx,
                chunk_begin,
            )
        else:
            comptime assert num_heads % 16 == 0, (
                "the scalar fp8_index_kernel tiles heads by thread_dim_y == 8"
                " and is unvalidated below 16 heads; num_heads in {4, 8}"
                " requires the SM100 tensor-core path"
            )
            comptime block_tile_shape: Array[Int, 2] = [512, 128]
            comptime BM = block_tile_shape[0]
            comptime BN = block_tile_shape[1]
            comptime smem_use = size_of[
                IndexSmemStorage[dtype, num_heads, depth, BN]
            ]()
            comptime smem_available = ctx.default_device_info.shared_memory_per_multiprocessor - 1024

            comptime kernel = fp8_index_kernel[
                dtype,
                type_of(scores_tile).LayoutType,
                type_of(q).LayoutType,
                type_of(q_s).LayoutType,
                type_of(k_operand),
                type_of(ks_operand),
                block_tile_shape,
                type_of(input_row_offsets.as_immut()).LayoutType,
                num_heads,
                depth,
            ]

            ctx.enqueue_function[kernel](
                scores_tile,
                q.as_immut(),
                q_s,
                k_operand,
                ks_operand,
                input_row_offsets.as_immut(),
                grid_dim=(
                    batch_size,
                    max_new_tokens,
                    ceildiv(max_num_keys, BM),
                ),
                block_dim=(16, 8, 1),
                shared_mem_bytes=smem_use,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_available)
                ),
            )

        # Apply mask for prefill (seq_len > 1). The SM100 scorer fuses the
        # causal mask into its store guard and the top-k reads only written
        # slots, so the separate full-buffer mask pass only runs for the scalar
        # fallback.
        comptime if mask_str != MaskName.NULL.name and not use_sm100_scorer:
            if max_new_tokens > 1:

                @always_inline
                def apply_mask_dispatch[
                    mask_t: MHAMask
                ](mask: mask_t) raises {imm}:
                    comptime mask_kernel = apply_mask_kernel[
                        mask_t,
                        scores_tile.LayoutType,
                        scores_tile.origin,
                        input_row_offsets.LayoutType,
                        ImmOrigin(input_row_offsets.origin),
                        type_of(cache_lengths).LayoutType,
                    ]

                    ctx.enqueue_function[mask_kernel](
                        scores_tile,
                        input_row_offsets.as_immut(),
                        cache_lengths,
                        mask,
                        Int32(max_num_keys),
                        grid_dim=(
                            batch_size,
                            ceildiv(max_new_tokens, 16),
                            ceildiv(max_num_keys, 16),
                        ),
                        block_dim=(16, 16, 1),
                    )

                dispatch_mask[mask_str](apply_mask_dispatch)

        # The bitonic path can only select up to the champion width
        # (PERSISTENT_TOPK_MAX_N); topk_gpu handles the rare k above it.
        if effective_k <= PERSISTENT_TOPK_MAX_N:
            persistent_topk_block_split(
                ctx,
                rebind[UnsafePointer[Float32, ImmutAnyOrigin]](scores_tile.ptr),
                rebind[UnsafePointer[Int32, MutAnyOrigin]](topk_idxs_tile.ptr)
                + chunk_begin * effective_k,
                max_num_keys,
                effective_k,
                chunk_rows,
                Optional(
                    rebind[UnsafePointer[Int32, ImmutAnyOrigin]](row_bounds_ptr)
                    + chunk_begin
                ),
            )
        else:
            # Only reachable unchunked (see `chunk_scores`), so the whole-batch
            # tiles below are this chunk.
            topk_gpu[sampling=False, largest=True](
                ctx,
                effective_k,
                scores_tile,
                topk_vals_tile,
                topk_idxs_tile,
            )

    # Fill invalid positions with -1:
    # - Positions [effective_k, top_k) when top_k > max_num_keys
    # - Positions where k_idx >= num_keys for that token (causal masking)
    comptime fill_kernel = fill_invalid_topk_kernel[
        input_row_offsets.LayoutType,
        ImmOrigin(input_row_offsets.origin),
        type_of(cache_lengths).LayoutType,
        use_causal_mask,
        kpool,
    ]

    var block_size = align_up(top_k, 32)
    block_size = min(block_size, 1024)  # Cap at max threads per block

    ctx.enqueue_function[fill_kernel](
        rebind[UnsafePointer[Int32, MutAnyOrigin]](output_indices.ptr),
        rebind[UnsafePointer[Int32, MutAnyOrigin]](topk_idxs_tile.ptr),
        input_row_offsets.as_immut(),
        cache_lengths,
        Int32(total_seq_len),
        Int32(top_k),
        Int32(effective_k),
        grid_dim=(total_seq_len, 1, 1),
        block_dim=(block_size, 1, 1),
    )

    _ = scores_buf
    _ = topk_vals_buf
    _ = topk_idxs_buf
    _ = row_bounds_buf
