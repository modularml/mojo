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
"""Graph-op bindings for the MiniMax-M3 block-sparse attention (MSA) kernels.

BF16, head_dim 128, paged KV with page_size 128 (== block size BN), single
index-K head.  Two ops:

  * `mo.msa.indexer.ragged.paged`   -> `sparse_indexer_{prefill,decode}`
  * `mo.msa.attention.ragged.paged` -> `msa_sm100_decode` (decode) /
    `msa_sm100_prefill_{plan,run}` (prefill) on NVIDIA SM100;
    `msa_amd_decode_dispatch` (decode) / `msa_amd_prefill_run` (prefill) on
    AMD gfx950 (MI355).

Both ops are dual-arch, so the dead arch's kernels never codegen (the SM100
tcgen05/TMA prefill cannot lower on gfx950, and the AMD MFMA path cannot lower
on SM100).  The attention op forks here, on `has_amd_gpu_accelerator()`; its AMD
prefill `plan` is the arch-neutral `msa_sm100_prefill_plan` (host sizing +
buffer alloc only -- no SM100 device kernel), shared by both arches, so only the
pure-device run differs.  The indexer op forks at both levels: `USE_AMD_MTP_SCORER`
picks the AMD-only speculative route *here*, because that route is a different
pair of kernels rather than a different lowering of the same one; every other
indexer route forks one level down, inside `sparse_indexer_{prefill,decode}`
and their `_mtp` siblings, which dispatch to the `*_amd` scorers and bitonic
top-k on gfx950.

Each op takes the same arguments for prefill and decode and picks the kernel at
runtime from `kv_collection.max_seq_length` (the max number of *new* query
tokens in the batch): `== 1` is a single-token decode step, `2-8` is
speculative decode, and anything larger is a prefill / context-encoding step.

The indexer op emits top-k *block* ids per (index head, token); the attention
op consumes those block ids (`d_indices`) to gather a sparse band of KV blocks
from the main paged cache.  Index-K and main-KV are each BF16 or scale-free
e4m3 (inferred from the operands).  Neither carries scales, so both build with
`generic_get_paged_cache`. The `msa_sm100_*` kernels accumulate in FP32.

Modeled on the MLA FP8 indexer registration in `attention.mojo`
(`mo.mla.indexer.ragged.float8.paged`) for the comptime cache-param extraction
+ paged-collection build, and on the in-tree MSA tests
(`Kernels/test/msa/test_msa_sm100_d128_decode_paged.mojo`,
`test_msa_d128_prefill_device_csr.mojo`) for the exact call shapes.

Three attention routes, picked at runtime from `kv_collection.max_seq_length`
(`max_q_len`, the max number of *new* query tokens; MAX draft length = 8):

  * `== 1`            -> single-token DECODE (`msa_sm100_decode`, NullMask, the
                        SM-fill split-K heuristic).  Causal is a no-op (the
                        single query sits at the sequence END, so every selected
                        past KV position is causal-valid and nothing is masked).
  * `2` through `8`    -> sparse SPECULATIVE decode (`msa_sm100_decode` on
                        NVIDIA, `msa_amd_decode_dispatch` on AMD) with
                        `spec_max_seq_len` bound to the matched length: one CTA
                        per (draft token, split-K partition), in-kernel
                        per-token causal, capture-stable over-launched grid
                        (`batch * spec_max_seq_len` on the token axis,
                        `max_num_partitions` on the partition axis).  At np>1
                        the partials key on the ragged global query row and the
                        shared `mha_splitk_reduce` combine writes the ragged
                        output directly.
                        Causal is REAL (a draft token can precede some selected
                        KV); the kernel derives each slot's logical KV start
                        in-kernel as `d_idx_base[blk] * BN` and each token's
                        logical query position as `cache_lengths[batch] +
                        tok_in_seq`, so the op never builds a `kv_logical_pos` or
                        `q_positions` array (mirrors the prefill `use_causal`
                        path, which derives the diagonal from cu_seqlens +
                        cache_lengths).  A short prefill of 2-8 is correctly
                        handled by this path, so no prefill/spec disambiguation
                        is needed.
  * `> 8`              -> PREFILL (`msa_sm100_prefill_{plan,run}`, device CSR).
"""

import extensibility

from std.collections import OptionalReg
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, min
from std.memory import UnsafePointer
from std.sys.info import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from layout import row_major, TileTensor, Coord
from layout.tile_tensor import row_major as tt_row_major

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor

from nn.kv_cache import generic_get_paged_cache
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import MHAConfig, StaticInt

from msa.sparse_indexer_prefill import sparse_indexer_prefill
from msa.sparse_indexer_decode import (
    _MMA_REG_MAX_ROWS,
    sparse_indexer_decode,
    sparse_indexer_decode_score_mtp,
    sparse_indexer_decode_topk_mtp,
)
from msa.msa_config import MSAConfig
from msa.msa_1q import msa_sm100_decode
from msa.msa_2q import msa_sm100_prefill_plan, msa_sm100_prefill_run
from msa.amd.decode import msa_amd_decode_dispatch
from msa.amd.prefill import msa_amd_prefill_run
from msa.amd.splitk_reduce_quant import MSA_MX_SF_VECTOR_SIZE

from linalg.block_scaled_quantization import quantize_mx_amd


# Widest draft window (`max_q_len`) the speculative-decode route accepts; above
# this both ops fall to prefill.  BOTH ops must read this same bound: the
# indexer capped lower would still emit a well-formed
# `[num_index_heads, total_q, topk]`, so attention would keep running
# speculative decode against prefill-scored blocks and the only symptom would be
# the per-call prefill plan this route exists to avoid.
comptime MAX_SPEC_DRAFT = 8


# ===-----------------------------------------------------------------------===#
# Indexer (top-k block selection)
# ===-----------------------------------------------------------------------===#


@always_inline
def _require_score_scratch[
    num_index_heads: Int
](
    score_scratch: MutableInputTensor[dtype=.float32, rank=3, ...],
    rows: Int,
    max_num_blocks: Int,
) raises:
    """Raise unless the scratch covers this single-token decode step.

    The multi-token routes test the same bounds in their route predicates and
    fall through to prefill, which allocates its own buffer. Single-token decode
    has no such fallback -- it IS the decode route -- so it raises instead.
    Nothing downstream would catch it: the kernels' own bound checks are
    `debug_assert`, so a release build would take the scorer's out-of-bounds
    write.

    Parameters:
        num_index_heads: Number of index (query) heads, for the message.

    Args:
        score_scratch: The caller's persistent scratch.
        rows: Score rows this step needs -- `batch` on single-token decode.
        max_num_blocks: This step's per-row block bound. Dim-2 is the row
            stride, so it only has to be at least this, not equal to it.
    """
    if (
        Int(score_scratch.dim_size[1]()) >= rows
        and Int(score_scratch.dim_size[2]()) >= max_num_blocks
    ):
        return
    raise Error(
        String(
            (
                "mo.msa.indexer.ragged.paged: score_scratch is too small for"
                " this decode step. Need at least ["
            ),
            num_index_heads,
            ", ",
            rows,
            ", ",
            max_num_blocks,
            "], got [",
            Int(score_scratch.dim_size[0]()),
            ", ",
            Int(score_scratch.dim_size[1]()),
            ", ",
            Int(score_scratch.dim_size[2]()),
            "].",
        )
    )


@extensibility.register("mo.msa.indexer.ragged.paged")
struct Struct_msa_indexer_ragged_paged:
    """Registers the `mo.msa.indexer.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        k_type: DType,
        //,
        *,
        num_index_heads: Int,
        idx_head_dim: Int,
        block_size: Int,
        topk: Int,
        init_blocks: Int,
        local_blocks: Int,
    ](
        out_idxs: OutputTensor[dtype=.int32, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        prefix_lens: InputTensor[dtype=.uint32, rank=1, ...],
        k_blocks: MutableInputTensor[dtype=k_type, rank=6, ...],
        k_cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        k_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        k_max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        k_max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        msa_scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        layer_idx: UInt32,
        score_scratch: MutableInputTensor[dtype=.float32, rank=3, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Select top-k key *blocks* per (index head, query token).

        Dispatches to the decode kernel when `kv_collection.max_seq_length == 1`
        (one new index-K token per sequence) and to the prefill kernel
        otherwise.

        Parameters:
            num_index_heads: Number of index (query) heads.
            idx_head_dim: Index head dimension.
            block_size: KV block size in tokens (== page_size).
            topk: Number of blocks to select per token.
            init_blocks: Always-keep leading blocks (forced high score).
            local_blocks: Always-keep trailing/local blocks (forced score).

        Args:
            out_idxs: Output block indices `[num_index_heads, num_rows, topk]`,
                int32, `-1`-padded (`num_rows` == total_q on prefill, batch on
                decode).
            q: Query tensor `[num_rows, num_index_heads, idx_head_dim]` BF16.
            input_row_offsets: Ragged query offsets `[batch + 1]` uint32 (used on
                the prefill path; on decode it is `[0, 1, ..., batch]`).
            prefix_lens: Per-batch cached-key count `[batch]` uint32 (pass the
                index-K `cache_lengths`); used as the decode `seq_lens`.
            k_blocks: Index-K paged blocks `[num_blocks, 1, num_layers,
                page_size, 1, idx_head_dim]`, BF16 or scale-free e4m3.
            k_cache_lengths: Index-K cache lengths `[batch]` uint32.
            k_lookup_table: Index-K page table `[batch, max_pages]` uint32.
            k_max_prompt_length: Index-K max prompt (query) length `[1]` uint32.
            k_max_cache_length: Index-K max cache length `[1]` uint32.
            msa_scalar_args: On-device scalar arguments for the decode indexer
                msa_scalar_args[0] = batch_size
                msa_scalar_args[1] = max_cache_valid_length.
            layer_idx: Layer index for the index-K cache.
            score_scratch: Persistent decode score scratch
                `[num_index_heads, max_rows, MAX_NUM_BLOCKS]`. `max_rows` must
                cover `total_q` for every step served: `max_batch` for
                single-token decode, and `max_batch * max_new_tokens_per_step`
                for the multi-token (MTP / speculative) routes below, which
                write at the ragged row `input_row_offsets[b] + t`. Those route
                predicates check this and fall back to prefill when the scratch
                is too narrow, so an old caller degrades rather than corrupts;
                single-token decode has no such fallback and raises instead.
            scale: QK scale.
            ctx: Device context.
        """
        comptime assert (
            k_type == DType.bfloat16 or k_type == DType.float8_e4m3fn
        ), "index-K cache must be bf16 or scale-free e4m3"
        var k_collection = generic_get_paged_cache(
            k_blocks,
            k_cache_lengths,
            k_lookup_table,
            k_max_prompt_length,
            k_max_cache_length,
        )
        var k_cache = k_collection.get_key_cache(Int(layer_idx))
        var k_operand = KVCacheMHAOperand(k_cache)

        var total_q = Int(q.dim_size[0]())
        if total_q == 0:
            return
        # AMD: `max_context_length()` is already post-write (it folds in this
        # step's query width), so it is the exact block count the split-K
        # decode top-k routes on -- re-adding `max_prompt_length()` would tip
        # `chunk_blocks` over the split-K `CHUNK_CAP` at a block-aligned top
        # context and misroute to the slow path.
        #
        # Post-write-ness is a property of the accessor and the host that fills
        # it, not of AMD, so NVIDIA could drop the term too. It keeps it for
        # now, but no longer for the reason first recorded here: SM100's decode
        # top-k is the bitonic arm, which DOES route on this value. The cost is
        # bounded and narrow -- the term adds at most one block, and the only
        # capacity where one block breaks split-K coverage is exactly 4096
        # (`cap_chunks` 16 -> 32), where the misroute lands on monolithic. That
        # fallback is not a cliff: ragged traffic crosses the 4096 boundary on
        # its own (capacity follows the batch's longest request), and on the
        # seeds where it does, monolithic still measures ~0.85 of the serial
        # arm this replaced. Dropping the term is a separate
        # change because it also shrinks the prefill scratch allocated below,
        # and because the unified-MTP draft steps substitute an in-graph bumped
        # `cache_lengths` for the host buffer while keeping the host
        # `max_cache_length` -- there this term is the only remaining slack, and
        # that bound is unproven.
        #
        # Safe because `max_context_length()` is >= every per-row post-write
        # context, so the kernel's own `num_blocks = ceildiv(seq_lens[b] +
        # in_step_q, block_size)` never exceeds this `max_num_blocks` (which also
        # sizes the `score` scratch); if that cache invariant broke, `score`
        # would be under-sized. The decode top-k kernels `debug_assert` it.
        var extra_keys = 0 if has_amd_gpu_accelerator() else Int(
            k_cache.max_prompt_length()
        )
        var max_num_blocks = ceildiv(
            Int(k_cache.max_context_length()) + extra_keys,
            block_size,
        )

        # The register-MMA decode scorer accepts a multi-token step when the
        # geometry is its own: NVIDIA, dim == block == 128, page-aligned,
        # nh <= MMA_M // 2. Mirrors `sparse_indexer_decode`'s own PER_TOKEN
        # comptime assert, so the elif body below is only elaborated where that
        # assert would pass. `topk` is unconstrained -- the MTP route selects
        # with the same unbounded `block_select_topk` as single-token decode.
        comptime MTP_DECODE_OK = (
            has_nvidia_gpu_accelerator()
            and idx_head_dim == 128
            and block_size == 128
            and (type_of(k_operand).page_size % block_size) == 0
            and num_index_heads <= _MMA_REG_MAX_ROWS // 2
        )
        var max_q_len = Int(k_collection.max_seq_length)

        # Decode == one new index-K token per sequence (`num_rows == batch`);
        # anything larger is a prefill / context-encoding step, EXCEPT a small
        # multi-token (MTP / speculative-decode) step, which the decode indexer
        # scores far more cheaply than prefill's MMA_M=128 machinery -- it packs
        # every (token, head) pair into ONE m16 fragment and reads K once. See
        # the MTP elif.
        if max_q_len == 1:
            var batch = total_q  # 1 token/seq on decode
            # Use persistent score scratch. This is required for graph capture.
            # One row per sequence here; the MTP route below needs one per
            # token.
            _require_score_scratch[num_index_heads](
                score_scratch, batch, max_num_blocks
            )
            var score = score_scratch.to_tile_tensor[.int64]()

            # `prefix_lens` is the index-K `cache_lengths` BEFORE this step's
            # IndexK was scattered (MAX `_is_cache_length_accurate=False`
            # convention). The decode scorer needs the POST-write key count, so
            # it adds this step's per-batch query-token count (from
            # `input_row_offsets`) internally. On the decode route there is
            # exactly one new token per sequence, so that count is 1; passing
            # `input_row_offsets` keeps it correct for any ragged in-step count.
            sparse_indexer_decode[
                DType.bfloat16,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q.to_tile_tensor[.int64](),
                k_operand,
                prefix_lens.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                score,
                out_idxs.to_tile_tensor[.int64](),
                batch,
                max_num_blocks,
                topk,
                init_blocks,
                local_blocks,
                scale,
                ctx,
            )
        elif (
            MTP_DECODE_OK
            # Rows must fit the m16 fragment, and the persistent scratch must
            # have a row per ragged output row. The scratch shape is a true
            # graph constant (fixed Python int at graph-build time).
            # `max_seq_length` is not -- it's a runtime scalar read from the
            # `k_max_prompt_length` graph input -- but the serving pipeline's
            # device-graph-capture runner pins it to `num_speculative_tokens +
            # 1` for every capture and replay of a bucket
            # (`ServeGraphCaptureRunner` in pipelines/lib/graph_capture.py),
            # and capture bakes this branch choice once per captured op
            # sequence, so this predicate is not re-evaluated on replay. A
            # scratch sized for single-token decode simply keeps the prefill
            # route rather than writing past its end.
            and num_index_heads * max_q_len <= _MMA_REG_MAX_ROWS
            and Int(score_scratch.dim_size[1]())
            >= (Int(input_row_offsets.dim_size[0]()) - 1) * max_q_len
        ):
            # ---- Small multi-token (MTP / speculative) decode step ----
            # Score each (token, head) pair -- `num_index_heads * max_q_len <=
            # 16` rows of one m16 fragment -- with its own Q-at-end causal
            # horizon and its own top-k. `out_idxs` and the persistent `score`
            # scratch use the ragged `[nh, total_q, ...]` prefill layout (row =
            # `input_row_offsets[b] + t`), so this is a drop-in for the same
            # sparse-attention consumer. Anything wider or non-layout-matching
            # keeps the prefill route below.
            #
            # This step is inside the graph-capture region, so it writes the
            # PERSISTENT scratch rather than allocating: the row-count predicate
            # above is what guarantees the ragged writes fit, and it degrades to
            # prefill instead of over-running a scratch sized for single-token
            # decode. The `comptime if` keeps the decode body from codegen'ing
            # where the geometry cannot support it.
            comptime if MTP_DECODE_OK:
                var batch = Int(input_row_offsets.dim_size[0]()) - 1
                var score = score_scratch.to_tile_tensor[.int64]()
                sparse_indexer_decode[
                    DType.bfloat16,
                    type_of(k_operand),
                    num_index_heads,
                    idx_head_dim,
                    block_size,
                    PER_TOKEN=True,
                ](
                    q.to_tile_tensor[.int64](),
                    k_operand,
                    prefix_lens.to_tile_tensor[.int64](),
                    input_row_offsets.to_tile_tensor[.int64](),
                    score,
                    out_idxs.to_tile_tensor[.int64](),
                    batch,
                    max_num_blocks,
                    topk,
                    init_blocks,
                    local_blocks,
                    scale,
                    ctx,
                    # Graph-constant new-token cap. Bounds every per-request
                    # `in_step_q`, so it is what specializes the scorer's row
                    # count and keeps it inside the m16 fragment.
                    max_q_len=max_q_len,
                )
        else:
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            # AMD speculative widths use the decode-style scorer and top-k;
            # width one retains the established decode route.  The scorer loads
            # each K vector once and reuses it across the compile-time query
            # width, so the whole draft window costs one pass over index-K.
            comptime USE_AMD_MTP_SCORER = (
                has_amd_gpu_accelerator()
                and num_index_heads == 1
                and idx_head_dim == 128
                and block_size == 128
                and type_of(k_operand).page_size % block_size == 0
            )
            comptime if USE_AMD_MTP_SCORER:
                var max_q_len = Int(k_collection.max_seq_length)
                # MTP indexes the score by GLOBAL QUERY ROW, so the scratch
                # needs `total_q` rows here, not the `batch` that single-token
                # decode needs. Testing that in the route predicate rather than
                # raising mirrors the NVIDIA MTP route above: too narrow a
                # scratch simply keeps prefill, which allocates its own buffer.
                #
                # Dim-2 is the row stride the kernels address with, while
                # `max_num_blocks` rides along as a separate runtime bound, so a
                # scratch cut for a longer context still serves a shorter step. A
                # layout rebuilt over the same memory with `max_num_blocks` as
                # the stride would misalign every row after the first.
                if (
                    2 <= max_q_len <= MAX_SPEC_DRAFT
                    and Int(score_scratch.dim_size[1]()) >= total_q
                    and Int(score_scratch.dim_size[2]()) >= max_num_blocks
                ):
                    var mtp_score = score_scratch.to_tile_tensor[.int64]()

                    comptime for query_width in range(2, MAX_SPEC_DRAFT + 1):
                        if max_q_len == query_width:
                            sparse_indexer_decode_score_mtp[
                                DType.bfloat16,
                                type_of(k_operand),
                                query_width,
                                num_index_heads,
                                idx_head_dim,
                                block_size,
                            ](
                                q.to_tile_tensor[.int64](),
                                k_operand,
                                prefix_lens.to_tile_tensor[.int64](),
                                input_row_offsets.to_tile_tensor[.int64](),
                                mtp_score,
                                batch,
                                total_q,
                                max_num_blocks,
                                init_blocks,
                                local_blocks,
                                scale,
                                ctx,
                            )
                            sparse_indexer_decode_topk_mtp[
                                query_width, num_index_heads, block_size
                            ](
                                prefix_lens.to_tile_tensor[.int64](),
                                input_row_offsets.to_tile_tensor[.int64](),
                                mtp_score,
                                out_idxs.to_tile_tensor[.int64](),
                                batch,
                                total_q,
                                max_num_blocks,
                                topk,
                                ctx,
                            )
                            return

            # Per-call score scratch, one row per TOKEN
            # `[num_index_heads, total_q, max_num_blocks]`. Prefill cannot use
            # the caller's `score_scratch`: `total_q` is a whole context window,
            # unbounded by any fixed allocation, and prefill is not
            # capture-sensitive.
            #
            # Deliberately left uninitialized. The scorer writes every block in
            # `[0, num_blocks)` of each row, and nothing reads the ragged tail
            # past that bound -- the top-k kernels recompute each row's block
            # count and clamp to it.
            var score_size = num_index_heads * total_q * max_num_blocks
            var score_buf = ctx.enqueue_create_buffer[.float32](score_size)
            var score = TileTensor(
                score_buf,
                tt_row_major(num_index_heads, total_q, max_num_blocks),
            )

            sparse_indexer_prefill[
                DType.bfloat16,
                type_of(k_operand),
                num_index_heads,
                idx_head_dim,
                block_size,
            ](
                q.to_tile_tensor[.int64](),
                k_operand,
                input_row_offsets.to_tile_tensor[.int64](),
                prefix_lens.to_tile_tensor[.int64](),
                score,
                out_idxs.to_tile_tensor[.int64](),
                batch,
                total_q,
                Int(k_cache.max_prompt_length()),  # max_seqlen_q
                max_num_blocks,
                topk,
                init_blocks,
                local_blocks,
                scale,
                ctx,
            )
            _ = score_buf^


# ===-----------------------------------------------------------------------===#
# Sparse attention (block-gathered MHA)
# ===-----------------------------------------------------------------------===#


@extensibility.register("mo.msa.attention.ragged.paged")
struct Struct_msa_attention_ragged_paged:
    """Registers the `mo.msa.attention.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        kv_type: DType,
        //,
        group: Int,
        topk: Int,
        sparse_block_size: Int,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=kv_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        cache_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        total_context_length: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        msa_scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        layer_idx: UInt32,
        d_indices: InputTensor[dtype=.int32, rank=3, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """Block-sparse MHA for SM100 (BF16 or FP8 e4m3, head_dim 128).

        The KV cache dtype (`kv_type`, inferred from `q` / `kv_blocks`) selects
        BF16 or native FP8 e4m3; `q` and `kv_blocks` must share it and the
        kernel accumulates in FP32. FP8 is scale-free (no per-block dequant
        scales), matching the `msa_sm100_*` FP8 path. The output is always BF16.

        Gathers `topk` KV blocks per (kv head, query token) using the block ids
        in `d_indices`.  Dispatches to the decode kernel when
        `kv_collection.max_seq_length == 1` (one query token per sequence) and to
        the prefill kernel otherwise.

        Decode uses `NullMask` + an SM-fill split-K heuristic
        (`get_mha_decoding_max_num_partitions` clamped by `topk`): `num_partitions
        > 1` runs the block-major fwd over partitioned KV bands then combines via
        the shared `mha_splitk_reduce`; `num_partitions == 1` takes the no-combine
        `NoPartition` path.  Prefill uses the
        device-CSR plan/run path (`msa_sm100_prefill_plan` +
        `msa_sm100_prefill_run`): the run is pure-device, but the plan sizes its
        buffers from the per-batch cu-seqlens on host, so one D2H readback +
        sync per call is unavoidable while this stays a single stateless op.

        Routing is purely by the runtime query length
        `max_q_len = kv_collection.max_seq_length` (the max new query tokens):
        `== 1` decode, `2` through `8` sparse speculative decode (one CTA per
        draft token, real per-token causal, capture-stable over-launch -- see the
        module docstring; `spec_max_seq_len` is bound to the matched length per
        branch), and `> 8` prefill.  A short 2-8 prefill is correctly handled by
        the spec path, so no prefill/spec disambiguation is needed.  Spec decode
        derives each draft token's logical query position in-kernel from
        `cache_lengths + tok_in_seq` (mirrors the prefill `use_causal` path), so
        no `q_positions` array is built or passed.

        Parameters:
            group: Query heads per kv-head (`n_heads // n_kv_heads`); asserts
                `group <= MMA_M` in the kernel.
            topk: Number of gathered KV blocks per token (`d_indices` stride).
            sparse_block_size: KV block size in tokens, from the model's
                `sparse_attention_config`. Must equal the KV cache
                `page_size` and the forward's `BN`; asserted below.

        Args:
            output: Output `[num_rows, n_heads, head_dim]` BF16.
            q: Query `[num_rows, n_heads, head_dim]`, dtype `kv_type` (BF16 or
                FP8 e4m3; `num_rows` == total_q on prefill, batch on decode).
            input_row_offsets: Ragged query offsets `[batch + 1]` uint32 (1
                token/seq on decode).
            cache_row_offsets: Ragged valid cache offsets `[batch + 1]` uint32.
            total_context_length: Total context length of the current batch.
            kv_blocks: Main-KV paged blocks `[num_blocks, 2, num_layers,
                page_size, n_kv_heads, head_dim]`, dtype `kv_type` (BF16 or
                FP8 e4m3, scale-free).
            cache_lengths: Main-KV cache lengths `[batch]` uint32.
            kv_lookup_table: Main-KV page table `[batch, max_pages]` uint32.
            max_prompt_length: Main-KV max prompt (query) length `[1]` uint32.
            max_cache_length: Main-KV max cache length `[1]` uint32.
            msa_scalar_args: On-device scalar arguments for the MSA decode
                msa_scalar_args[0] = batch_size
                msa_scalar_args[1] = max_cache_valid_length.
            layer_idx: Layer index for the main-KV cache.
            d_indices: Selected block ids `[n_kv_heads, num_rows, topk]` int32.
            scale: QK scale.
            ctx: Device context.
        """
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var k_cache = kv_collection.get_key_cache(Int(layer_idx))
        var v_cache = kv_collection.get_value_cache(Int(layer_idx))
        var k_op = KVCacheMHAOperand(k_cache)
        var v_op = KVCacheMHAOperand(v_cache)

        comptime k_num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime num_heads = group * k_num_heads
        # The selection granularity must equal the KV cache page size:
        # one KV tile == one page == one sparse block.  Asserted here
        # because a mismatch has no fault to catch downstream:
        # `k2q_csr_sizes`' bounds stop holding and the block gather
        # silently drops selections the indexer made.
        comptime assert (
            sparse_block_size == page_size
        ), "sparse_block_size must equal the KV cache page_size"
        comptime msa = MSAConfig(block_size=sparse_block_size, topk=topk)
        comptime config = MHAConfig[kv_type](num_heads, head_dim)

        # `num_rows` == total query tokens (== batch on decode, 1 token/seq).
        var num_rows = Int(q.dim_size[0]())

        # A data-parallel replica with no assigned requests gets empty per-rank
        # inputs. There is nothing to attend, and the routes below build a Q TMA
        # descriptor, which rejects a zero global dim.
        if num_rows == 0:
            return

        # Non-owning DeviceBuffer views over the graph tensors.
        var out_lt = output.to_layout_tensor()
        var q_lt = q.to_layout_tensor()
        var output_buf = DeviceBuffer[.bfloat16](
            ctx, out_lt.ptr, num_rows * num_heads * head_dim, owning=False
        )
        var q_buf = DeviceBuffer[kv_type](
            ctx, q_lt.ptr, num_rows * num_heads * head_dim, owning=False
        )

        # Route purely on the runtime query length.  Both architectures share
        # the module-level `MAX_SPEC_DRAFT`; larger query lengths use prefill.
        var max_q_len = Int(kv_collection.max_seq_length)

        # Decode == one query token per sequence (`max_q_len == 1`).
        if max_q_len == 1:
            var topk_tokens = topk * page_size

            var iro_lt = input_row_offsets.to_layout_tensor()
            var valid_length = DeviceBuffer[.uint32](
                ctx,
                iro_lt.ptr,
                Int(input_row_offsets.dim_size[0]()),
                owning=False,
            )
            var d_indices_tt = TileTensor(
                d_indices.to_layout_tensor().ptr,
                row_major(Coord(d_indices.to_layout_tensor().size())),
            ).as_immut()

            # `np` is owned by the decode entry (computed from batch_size,
            # topk_tokens, topk via the dense-MHA heuristic).
            #
            # `mask_unselected=True`: the indexer `-1`-pads `d_indices` when
            # the sequence has fewer than `topk` selectable blocks (e.g. a
            # short first decode step). Without this the `-1` blocks attend
            # phantom rows, and -- with np == topk -- each `-1` lands in its
            # own fully-masked partition whose NaN exp-sum poisons the combine.
            #
            # AMD and SM100 take the same call signature; the two arms differ
            # only in kernel name.  No `valid_key` (the indexer's trailing
            # block is whole; a sub-BN partial would need per-batch clamp).
            comptime if has_amd_gpu_accelerator():
                msa_amd_decode_dispatch[
                    config=config,
                    group=group,
                    ragged=True,
                    _is_cache_length_accurate=False,
                    mask_unselected=True,
                ](
                    output_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_tt,
                    topk,  # indices_stride (topk in BLOCKS)
                    num_rows,  # num_rows_q (1 token/seq)
                    NullMask(),
                    valid_length,
                    StaticInt[1](),  # max_prompt_len (decode)
                    topk_tokens,  # max_cache_valid_length
                    scale,
                    None,  # kv_input_row_offsets
                    num_rows,  # batch_size
                    ctx,
                )
            else:
                msa_sm100_decode[
                    config=config,
                    group=group,
                    ragged=True,
                    _is_cache_length_accurate=False,
                    mask_unselected=True,
                ](
                    output_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_tt,
                    Int32(topk),  # indices_stride (topk in BLOCKS)
                    Int32(num_rows),  # num_rows_q (1 token/seq)
                    NullMask(),
                    valid_length,
                    StaticInt[1](),  # max_prompt_len (decode)
                    Int32(topk_tokens),  # max_cache_valid_length
                    scale,
                    None,  # kv_input_row_offsets
                    Int32(num_rows),  # batch_size
                    ctx,
                )
        elif 1 < max_q_len <= MAX_SPEC_DRAFT:
            # ---- Sparse SPECULATIVE decode ------------------------------
            # Each draft token runs on its OWN CTA via the per-token decode
            # kernel (`spec_max_seq_len > 1` derives the spec mode in-entry =>
            # per_token_index + causal + the over-launched
            # `batch * spec_max_seq_len` grid).  Selection reuses the PREFILL
            # indexer (per-token `[head_kv, total_q, topk]`), so the block ids
            # here are per draft token.  `input_row_offsets` is the ragged Q
            # offset array the kernel reads for the over-launch token tail and
            # the global-query-row remap (`iro[b] + tok_in_seq`).  Causal is
            # REAL here (a draft token can precede some selected KV): the kernel
            # poisons slots whose logical position exceeds the token's logical
            # query position, deriving the slot's logical start in-kernel from
            # `d_idx_base[blk]*BN` (no `kv_logical_pos` array) and the token's
            # logical query position in-kernel from
            # `cache_lengths[batch_of_token] + tok_in_seq` (no `q_positions`
            # array -- mirrors the prefill `use_causal` path, which derives the
            # diagonal from cu_seqlens + cache_lengths).  REAL split-K:
            # Both architecture entries feed `batch * spec_max_seq_len` to the
            # decode partition heuristic and key partials on the packed query
            # row, so the shared combine writes ragged output directly.
            var iro_lt = input_row_offsets.to_layout_tensor()
            var valid_length = DeviceBuffer[.uint32](
                ctx,
                iro_lt.ptr,
                Int(input_row_offsets.dim_size[0]()),
                owning=False,
            )
            var d_indices_tt = TileTensor(
                d_indices.to_layout_tensor().ptr,
                row_major(Coord(d_indices.to_layout_tensor().size())),
            ).as_immut()
            var topk_tokens = topk * page_size
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            # The over-launch span is a graph constant, so bind it to the
            # matched runtime length per branch.
            comptime for n in range(2, MAX_SPEC_DRAFT + 1):
                if max_q_len == n:
                    comptime if has_amd_gpu_accelerator():
                        msa_amd_decode_dispatch[
                            config=config,
                            group=group,
                            ragged=True,
                            _is_cache_length_accurate=False,
                            mask_unselected=True,
                            spec_max_seq_len=n,
                        ](
                            output_buf,
                            q_buf,
                            k_op,
                            v_op,
                            d_indices_tt,
                            topk,
                            num_rows,
                            NullMask(),
                            valid_length,
                            StaticInt[1](),
                            topk_tokens,
                            scale,
                            None,
                            batch,
                            ctx,
                        )
                    else:
                        msa_sm100_decode[
                            config=config,
                            group=group,
                            ragged=True,
                            _is_cache_length_accurate=False,
                            mask_unselected=True,
                            spec_max_seq_len=n,  # over-launch span (graph const)
                        ](
                            output_buf,
                            q_buf,
                            k_op,
                            v_op,
                            d_indices_tt,
                            Int32(topk),  # indices_stride (topk in BLOCKS)
                            Int32(num_rows),  # num_rows_q (total draft tokens)
                            NullMask(),
                            valid_length,  # ragged Q offsets (tail + row remap)
                            StaticInt[
                                1
                            ](),  # max_prompt_len: tile is decode-shaped
                            Int32(topk_tokens),  # max_cache_valid_length
                            scale,
                            None,  # kv_input_row_offsets
                            Int32(
                                batch
                            ),  # batch_size (grid.x = batch*spec_max_seq_len)
                            ctx,
                            # Spec decode derives BOTH the per-block logical
                            # start and the per-token logical query position
                            # in-kernel (the latter from `cache_lengths +
                            # tok_in_seq`), so it carries neither a
                            # `kv_logical_pos` nor a `q_positions` array.  The
                            # kernel keys causal off the derived spec mode (=>
                            # `causal`), not off the presence of a `q_positions`
                            # pointer.
                        )
                    return
        else:
            var batch = Int(input_row_offsets.dim_size[0]()) - 1

            var lse_buf = ctx.enqueue_create_buffer[.float32](
                num_rows * num_heads
            )

            var d_lt = d_indices.to_layout_tensor()
            var d_indices_buf = DeviceBuffer[.int32](
                ctx, d_lt.ptr, k_num_heads * num_rows * topk, owning=False
            )

            var plan = msa_sm100_prefill_plan[
                output_type=DType.bfloat16,
                config=config,
                group=group,
                topk=topk,
                msa=msa,
            ](
                num_rows,
                Int(total_context_length[0]),
                batch,
                Int(kv_collection.max_seq_length),
                Int(kv_collection.max_cache_length),
                ctx,
            )

            # bitcast input_row_offsets and cache_row_offsets to int32, then
            # wrap then in DeviceBuffer.
            var cuq_d = DeviceBuffer[.int32](
                ctx,
                input_row_offsets._ptr.bitcast[Int32](),
                batch + 1,
                owning=False,
            )
            var cuk_d = DeviceBuffer[.int32](
                ctx,
                cache_row_offsets._ptr.bitcast[Int32](),
                batch + 1,
                owning=False,
            )

            # The plan (host sizing + buffer alloc) is arch-neutral (no SM100
            # device kernel), so it codegens on gfx950.  Only the device run
            # differs: AMD chains CSR-build -> block-major fwd -> combine;
            # SM100 runs its tcgen05 path.  Both take the identical signature;
            # the comptime branch keeps the dead arch's kernels from codegen'ing.
            comptime if has_amd_gpu_accelerator():
                msa_amd_prefill_run[
                    config=config,
                    group=group,
                    topk=topk,
                    use_causal=True,
                ](
                    plan,
                    output_buf,
                    lse_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_buf,
                    cuq_d,
                    cuk_d,
                    scale,
                    ctx,
                )
            else:
                msa_sm100_prefill_run[
                    config=config,
                    group=group,
                    topk=topk,
                    use_causal=True,
                ](
                    plan,
                    output_buf,
                    lse_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_buf,
                    cuq_d,
                    cuk_d,
                    scale,
                    ctx,
                )

            _ = lse_buf^


@extensibility.register("mo.msa.attention.ragged.paged.mxfp8")
struct Struct_msa_attention_ragged_paged_mxfp8:
    """Registers the `mo.msa.attention.ragged.paged.mxfp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        kv_type: DType,
        //,
        group: Int,
        topk: Int,
        sparse_block_size: Int,
    ](
        output: OutputTensor[dtype=.float8_e4m3fn, rank=3, ...],
        output_scales: OutputTensor[dtype=.float8_e8m0fnu, rank=2, ...],
        q: InputTensor[dtype=kv_type, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        cache_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        total_context_length: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        msa_scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        layer_idx: UInt32,
        d_indices: InputTensor[dtype=.int32, rank=3, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """`mo.msa.attention.ragged.paged` that emits MXFP8 + E8M0 block scales.

        AMD (gfx950) only. Same inputs and routing as the BF16 op above, but
        the output is the o_proj-ready MXFP8 activation: `output` is
        `float8_e4m3fn` `[num_rows, n_heads, head_dim]` and `output_scales` is
        `float8_e8m0fnu` `[num_rows, n_heads * head_dim / 32]`, row-major --
        exactly what `quantize_mx_amd` produces and
        `mo.matmul.dynamic.block.scaled.amd` consumes. Bit-identical to running
        the BF16 op followed by that quantize (KERN-3384).

        Only the split-K decode/spec route saves a dispatch: the combine and
        the quantize fuse into `msa_amd_splitk_reduce_quant_mx`. Prefill and
        the `num_partitions <= 1` decode shapes still produce BF16 first (into
        a scratch buffer the fused route never touches) and quantize with the
        stock `quantize_mx_amd` -- the same two dispatches those routes cost
        unfused, so the op's output contract is uniform across routes.

        Deliberately a separate registration rather than a second output on
        the BF16 op: that op serves both vendors and BF16-o_proj configs, and
        this one only exists where o_proj consumes MXFP8. The routing below is
        the AMD half of the BF16 op's; a route added there needs a mirror
        here.

        Parameters:
            group: Query heads per kv-head (`n_heads // n_kv_heads`).
            topk: Number of gathered KV blocks per token (`d_indices` stride).
            sparse_block_size: KV block size in tokens, from the model's
                `sparse_attention_config`. Must equal the KV cache
                `page_size` and the forward's `BN`; asserted below.

        Args:
            output: Quantized output `[num_rows, n_heads, head_dim]` FP8 e4m3.
            output_scales: E8M0 block scales `[num_rows, n_heads * head_dim /
                32]`.
            q: Query `[num_rows, n_heads, head_dim]`, dtype `kv_type`.
            input_row_offsets: Ragged query offsets `[batch + 1]` uint32.
            cache_row_offsets: Ragged valid cache offsets `[batch + 1]` uint32.
            total_context_length: Total context length of the current batch.
            kv_blocks: Main-KV paged blocks `[num_blocks, 2, num_layers,
                page_size, n_kv_heads, head_dim]`, dtype `kv_type`.
            cache_lengths: Main-KV cache lengths `[batch]` uint32.
            kv_lookup_table: Main-KV page table `[batch, max_pages]` uint32.
            max_prompt_length: Main-KV max prompt (query) length `[1]` uint32.
            max_cache_length: Main-KV max cache length `[1]` uint32.
            msa_scalar_args: On-device scalar arguments (parity with the BF16
                op).
            layer_idx: Layer index for the main-KV cache.
            d_indices: Selected block ids `[n_kv_heads, num_rows, topk]` int32.
            scale: QK scale.
            ctx: Device context.
        """
        comptime if not has_amd_gpu_accelerator():
            raise Error(
                "mo.msa.attention.ragged.paged.mxfp8 is implemented for AMD"
                " gfx950 only; use mo.msa.attention.ragged.paged elsewhere"
            )
        else:
            var kv_collection = generic_get_paged_cache(
                kv_blocks,
                cache_lengths,
                kv_lookup_table,
                max_prompt_length,
                max_cache_length,
            )
            var k_cache = kv_collection.get_key_cache(Int(layer_idx))
            var v_cache = kv_collection.get_value_cache(Int(layer_idx))
            var k_op = KVCacheMHAOperand(k_cache)
            var v_op = KVCacheMHAOperand(v_cache)

            comptime k_num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
            comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
            comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
            comptime num_heads = group * k_num_heads
            # The selection granularity must equal the KV cache page size:
            # one KV tile == one page == one sparse block.  Asserted here
            # because a mismatch has no fault to catch downstream:
            # `k2q_csr_sizes`' bounds stop holding and the block gather
            # silently drops selections the indexer made.
            comptime assert (
                sparse_block_size == page_size
            ), "sparse_block_size must equal the KV cache page_size"
            comptime msa = MSAConfig(block_size=sparse_block_size, topk=topk)
            comptime config = MHAConfig[kv_type](num_heads, head_dim)
            comptime row_width = num_heads * head_dim
            comptime scale_cols = row_width // MSA_MX_SF_VECTOR_SIZE
            comptime assert (
                row_width % MSA_MX_SF_VECTOR_SIZE == 0
            ), "a scale block must not straddle rows"

            var num_rows = Int(q.dim_size[0]())
            if num_rows == 0:
                return

            var out_lt = output.to_layout_tensor()
            var scales_lt = output_scales.to_layout_tensor()
            var q_lt = q.to_layout_tensor()
            var mx_output_buf = DeviceBuffer[.float8_e4m3fn](
                ctx, out_lt.ptr, num_rows * row_width, owning=False
            )
            var mx_scales_buf = DeviceBuffer[.float8_e8m0fnu](
                ctx, scales_lt.ptr, num_rows * scale_cols, owning=False
            )
            var q_buf = DeviceBuffer[kv_type](
                ctx, q_lt.ptr, num_rows * row_width, owning=False
            )

            var max_q_len = Int(kv_collection.max_seq_length)

            if max_q_len == 1:
                var topk_tokens = topk * page_size
                var iro_lt = input_row_offsets.to_layout_tensor()
                var valid_length = DeviceBuffer[.uint32](
                    ctx,
                    iro_lt.ptr,
                    Int(input_row_offsets.dim_size[0]()),
                    owning=False,
                )
                var d_indices_tt = TileTensor(
                    d_indices.to_layout_tensor().ptr,
                    row_major(Coord(d_indices.to_layout_tensor().size())),
                ).as_immut()

                msa_amd_decode_dispatch[
                    config=config,
                    group=group,
                    ragged=True,
                    _is_cache_length_accurate=False,
                    mask_unselected=True,
                    quantize_mx=True,
                ](
                    # No BF16 landing buffer: the fused split-K route writes
                    # the FP8 outputs directly, and the np<=1 route allocates
                    # its own inside the dispatch. Spelled with the element
                    # type because it is what infers the dispatch's
                    # `output_type`.
                    Optional[DeviceBuffer[.bfloat16]](),
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_tt,
                    topk,  # indices_stride (topk in BLOCKS)
                    num_rows,  # num_rows_q (1 token/seq)
                    NullMask(),
                    valid_length,
                    StaticInt[1](),  # max_prompt_len (decode)
                    topk_tokens,  # max_cache_valid_length
                    scale,
                    None,  # kv_input_row_offsets
                    num_rows,  # batch_size
                    ctx,
                    mx_output=mx_output_buf,
                    mx_scales=mx_scales_buf,
                )
            elif 1 < max_q_len <= MAX_SPEC_DRAFT:
                var iro_lt = input_row_offsets.to_layout_tensor()
                var valid_length = DeviceBuffer[.uint32](
                    ctx,
                    iro_lt.ptr,
                    Int(input_row_offsets.dim_size[0]()),
                    owning=False,
                )
                var d_indices_tt = TileTensor(
                    d_indices.to_layout_tensor().ptr,
                    row_major(Coord(d_indices.to_layout_tensor().size())),
                ).as_immut()
                var topk_tokens = topk * page_size
                var batch = Int(input_row_offsets.dim_size[0]()) - 1

                comptime for n in range(2, MAX_SPEC_DRAFT + 1):
                    if max_q_len == n:
                        msa_amd_decode_dispatch[
                            config=config,
                            group=group,
                            ragged=True,
                            _is_cache_length_accurate=False,
                            mask_unselected=True,
                            spec_max_seq_len=n,
                            quantize_mx=True,
                        ](
                            Optional[DeviceBuffer[.bfloat16]](),
                            q_buf,
                            k_op,
                            v_op,
                            d_indices_tt,
                            topk,
                            num_rows,
                            NullMask(),
                            valid_length,
                            StaticInt[1](),
                            topk_tokens,
                            scale,
                            None,
                            batch,
                            ctx,
                            mx_output=mx_output_buf,
                            mx_scales=mx_scales_buf,
                        )
                        return
            else:
                var batch = Int(input_row_offsets.dim_size[0]()) - 1

                var lse_buf = ctx.enqueue_create_buffer[.float32](
                    num_rows * num_heads
                )

                var d_lt = d_indices.to_layout_tensor()
                var d_indices_buf = DeviceBuffer[.int32](
                    ctx, d_lt.ptr, k_num_heads * num_rows * topk, owning=False
                )

                # Prefill reduces into BF16 first; only this route pays for
                # the landing buffer.
                var bf16_scratch = ctx.enqueue_create_buffer[.bfloat16](
                    num_rows * row_width
                )

                var plan = msa_sm100_prefill_plan[
                    output_type=DType.bfloat16,
                    config=config,
                    group=group,
                    topk=topk,
                    msa=msa,
                ](
                    num_rows,
                    Int(total_context_length[0]),
                    batch,
                    Int(kv_collection.max_seq_length),
                    Int(kv_collection.max_cache_length),
                    ctx,
                )

                var cuq_d = DeviceBuffer[.int32](
                    ctx,
                    input_row_offsets._ptr.bitcast[Int32](),
                    batch + 1,
                    owning=False,
                )
                var cuk_d = DeviceBuffer[.int32](
                    ctx,
                    cache_row_offsets._ptr.bitcast[Int32](),
                    batch + 1,
                    owning=False,
                )

                msa_amd_prefill_run[
                    config=config,
                    group=group,
                    topk=topk,
                    use_causal=True,
                ](
                    plan,
                    bf16_scratch,
                    lse_buf,
                    q_buf,
                    k_op,
                    v_op,
                    d_indices_buf,
                    cuq_d,
                    cuk_d,
                    scale,
                    ctx,
                )

                quantize_mx_amd(
                    ctx,
                    TileTensor(
                        mx_output_buf.unsafe_ptr(),
                        row_major(num_rows, row_width),
                    ),
                    TileTensor(
                        mx_scales_buf.unsafe_ptr(),
                        row_major(num_rows, scale_cols),
                    ),
                    TileTensor(
                        bf16_scratch.unsafe_ptr(),
                        row_major(num_rows, row_width),
                    ),
                )

                _ = lse_buf^
                _ = bf16_scratch^
