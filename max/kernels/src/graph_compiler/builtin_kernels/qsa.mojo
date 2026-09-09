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
"""Graph-op bindings for the QSA lightning indexer and its attention consumer.

The kernels live in `//Kernels/lib/qsa` (`qsa.compress_keys`,
`qsa.block_score`, `qsa.sparse_attention`); only the
`@extensibility.register` wrappers are here, because a registration has to be
declared inside the built-in kernel library for a served graph to resolve the
op. Same shape as the `attn_res.mojo` and `msa.mojo` bindings.
"""

import extensibility

from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor
from max.gpu.host import DeviceContext

from nn.attention.mha_operand import KVCacheMHAOperand
from nn.kv_cache import generic_get_paged_cache
from qsa.block_score import qsa_block_score
from qsa.compress_keys import qsa_compress_keys
from qsa.sparse_attention import qsa_sparse_attention


@extensibility.register("qsa_block_score")
struct QSABlockScore:
    """Relu-summed, block-causally masked QSA block scores.

    Tensor shapes:
        - scores            : [queries, out_cols]                    (OUT, f32)
        - q                 : [queries, num_heads, head_dim]
        - block_keys        : [batch * max_blocks, head_dim]
        - query_positions   : [queries]                              (int32)
        - input_row_offsets : [batch + 1]                            (uint32)

    `block_keys` holds each sequence's blocks at the uniform stride
    `max_blocks = rows // batch`, which is exactly what `qsa_compress_keys`
    writes -- score column `j` is sequence-local block `j`, so a selection
    expands to sequence-local token indices for the consumer.

    `out_cols` must be at least `max_blocks`; the surplus is filled with `-inf`
    so a consumer can block-top-k a fixed-width row even when the live block
    count is below its `k`.
    """

    @staticmethod
    def execute[
        dtype: DType,
        target: StaticString,
        num_heads: Int,
        head_dim: Int,
        ratio: Int,
        score_scale: StaticString,
    ](
        scores: OutputTensor[dtype=DType.float32, rank=2, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        block_keys: InputTensor[dtype=dtype, rank=2, ...],
        query_positions: InputTensor[dtype=DType.int32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        if q.dim_size(1) != num_heads or q.dim_size(2) != head_dim:
            raise Error(
                "qsa_block_score: q shape disagrees with num_heads/head_dim"
            )
        if block_keys.dim_size(1) != head_dim:
            raise Error(
                "qsa_block_score: block_keys width disagrees with head_dim"
            )

        # `ops.custom`'s extensibility bridge takes no float parameters, so the
        # divisor -- `sqrt(indexer_head_dim)`, settled by `transformers`,
        # SGLang and vLLM -- arrives string-encoded from the call site rather
        # than being re-derived here, so the two cannot drift.
        var inv_scale = Float32(1.0) / Float32(atof(score_scale))

        var scores_tt = scores.to_tile_tensor[.int64]()
        var q_tt = q.to_tile_tensor[.int64]()
        var keys_tt = block_keys.to_tile_tensor[.int64]()
        var pos_tt = query_positions.to_tile_tensor[.int64]()
        var iro_tt = input_row_offsets.to_tile_tensor[.int64]()

        qsa_block_score[
            dtype,
            scores_tt.LayoutType,
            scores_tt.Engine,
            q_tt.LayoutType,
            q_tt.Engine,
            keys_tt.LayoutType,
            keys_tt.Engine,
            pos_tt.LayoutType,
            pos_tt.Engine,
            iro_tt.LayoutType,
            iro_tt.Engine,
            num_heads,
            head_dim,
            ratio,
            target,
        ](scores_tt, q_tt, keys_tt, pos_tt, iro_tt, inv_scale, ctx)


@extensibility.register("mo.qsa.attention.ragged.paged")
struct QSASparseAttentionRaggedPaged:
    """Gather-GQA over the token indices a QSA selection names.

    Tensor shapes:
        - output            : [total_q, num_q_heads, head_dim]        (OUT)
        - q                 : [total_q, num_q_heads, head_dim]
        - input_row_offsets : [batch + 1]                             (uint32)
        - kv_blocks         : [num_pages, 2, num_layers, page_size,
                              num_kv_heads, head_dim]                (mut in)
        - cache_lengths     : [batch]                                 (uint32)
        - kv_lookup_table   : [batch, max_pages]                      (uint32)
        - max_prompt_length : [1]                                     (uint32)
        - max_cache_length  : [1]                                     (uint32)
        - token_indices     : [total_q, selection_width]              (int32)
        - counts            : [total_q]                               (int32)

    `token_indices` holds positions within each query's own sequence, valid
    entries first; entries at or past `counts` are never read.
    """

    @staticmethod
    def execute[
        dtype: DType,
        out_dtype: DType,
        target: StaticString,
        group: Int,
        threads: Int,
        unroll: Int,
    ](
        output: OutputTensor[dtype=out_dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        token_indices: InputTensor[dtype=DType.int32, rank=2, ...],
        counts: InputTensor[dtype=DType.int32, rank=1, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) capturing raises:
        """Runs `qsa_sparse_attention` for one layer.

        Parameters:
            dtype: Element dtype of `q` and the KV cache (inferred).
            out_dtype: Element dtype of `output` (inferred).
            target: Compilation target.
            group: Query heads per kv head.
            threads: Threads per CTA; must satisfy the kernel's divisibility
                asserts.
            unroll: Gathered keys loaded before any is consumed.

        Args:
            output: Attention output.
            q: Queries.
            input_row_offsets: Ragged query offsets.
            kv_blocks: Paged KV blocks.
            cache_lengths: Per-sequence cached-key count.
            kv_lookup_table: Per-sequence page table.
            max_prompt_length: Max new query tokens this step.
            max_cache_length: Max cached context this step.
            layer_idx: Layer index into the KV cache.
            token_indices: Selected positions per query row.
            counts: Valid entries per query row.
            scale: QK scale.
            ctx: Device context.

        Raises:
            Error: If the operand shapes disagree with the parameters.
        """
        comptime num_kv_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime num_q_heads = group * num_kv_heads

        if q.dim_size(1) != num_q_heads or q.dim_size(2) != head_dim:
            raise Error(
                "qsa_sparse_attention: q shape disagrees with the cache's"
                " head geometry"
            )
        if output.dim_size(1) != num_q_heads or output.dim_size(2) != head_dim:
            raise Error("qsa_sparse_attention: output shape disagrees with q")

        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var k_op = KVCacheMHAOperand(
            kv_collection.get_key_cache(Int(layer_idx))
        )
        var v_op = KVCacheMHAOperand(
            kv_collection.get_value_cache(Int(layer_idx))
        )

        var out_tt = output.to_tile_tensor[.int64]()
        var q_tt = q.to_tile_tensor[.int64]()
        var idx_tt = token_indices.to_tile_tensor[.int64]()
        var cnt_tt = counts.to_tile_tensor[.int64]()
        var iro_tt = input_row_offsets.to_tile_tensor[.int64]()

        qsa_sparse_attention[
            dtype,
            out_dtype,
            out_tt.LayoutType,
            out_tt.Engine,
            q_tt.LayoutType,
            q_tt.Engine,
            idx_tt.LayoutType,
            idx_tt.Engine,
            cnt_tt.LayoutType,
            cnt_tt.Engine,
            iro_tt.LayoutType,
            iro_tt.Engine,
            type_of(k_op),
            type_of(v_op),
            num_q_heads,
            head_dim,
            group,
            threads,
            unroll,
            target,
        ](
            out_tt,
            q_tt,
            idx_tt,
            cnt_tt,
            iro_tt,
            k_op,
            v_op,
            scale,
            ctx,
        )


@extensibility.register("mo.qsa.compress_keys.paged")
struct QSACompressKeysPaged:
    """Pools a paged group of raw QSA index keys into dense block keys.

    Tensor shapes:
        - block_keys        : [batch * max_blocks, head_dim]           (OUT)
        - gamma             : [head_dim]
        - freqs_cis         : [positions, rotary_dim]
        - key_counts        : [batch]                                 (int32)
        - kv_blocks         : [num_pages, 2, num_layers, page_size,
                              1, head_dim]                          (mut in)
        - cache_lengths     : [batch]                                 (uint32)
        - kv_lookup_table   : [batch, max_pages]                      (uint32)
        - max_prompt_length : [1]                                     (uint32)
        - max_cache_length  : [1]                                     (uint32)

    `key_counts` is each sequence's total raw-key count *including* this
    forward's tokens, so it is `cache_lengths + this step's row lengths` rather
    than either alone. `max_blocks` is `block_keys.dim(0) // batch`, the same
    per-sequence stride `qsa_block_score` re-derives.
    """

    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        target: StaticString,
        head_dim: Int,
        rotary_dim: Int,
        ratio: Int,
        eps: StaticString,
    ](
        block_keys: OutputTensor[dtype=dtype, rank=2, ...],
        gamma: InputTensor[dtype=dtype, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        key_counts: InputTensor[dtype=DType.int32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=DType.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=DType.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=DType.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) capturing raises:
        """Runs `qsa_compress_keys` for one layer.

        Parameters:
            dtype: Element dtype of the keys, `gamma` and the output (inferred).
            freq_dtype: Element dtype of `freqs_cis` (inferred).
            target: Compilation target.
            head_dim: `indexer_head_dim`.
            rotary_dim: Channels the rotation covers.
            ratio: `indexer_compress_ratio`.
            eps: `rms_norm_eps`, string-encoded because the extensibility
                bridge takes no float parameters.

        Args:
            block_keys: Dense block keys, at the per-sequence stride.
            gamma: `k_layernorm.weight`.
            freqs_cis: Rotary cos/sin-pair table, indexed by position.
            key_counts: Raw keys held per sequence, this step included.
            kv_blocks: Paged blocks of the indexer cache group.
            cache_lengths: Per-sequence cached-key count.
            kv_lookup_table: Per-sequence page table.
            max_prompt_length: Max new query tokens this step.
            max_cache_length: Max cached context this step.
            layer_idx: Layer index into the indexer cache.
            ctx: Device context.

        Raises:
            Error: If the operand shapes disagree with the parameters.
        """
        comptime cache_kv_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime cache_head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime assert cache_kv_heads == 1, (
            "QSA scores one shared key head; the indexer cache group must have"
            " exactly one kv head"
        )
        comptime assert (
            cache_head_dim == head_dim
        ), "the indexer cache's head_dim must be indexer_head_dim"

        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var k_op = KVCacheMHAOperand(
            kv_collection.get_key_cache(Int(layer_idx))
        )

        var out_tt = block_keys.to_tile_tensor[.int64]()
        var gamma_tt = gamma.to_tile_tensor[.int64]()
        var freqs_tt = freqs_cis.to_tile_tensor[.int64]()
        var counts_tt = key_counts.to_tile_tensor[.int64]()

        qsa_compress_keys[
            dtype,
            freq_dtype,
            out_tt.LayoutType,
            out_tt.Engine,
            gamma_tt.LayoutType,
            gamma_tt.Engine,
            freqs_tt.LayoutType,
            freqs_tt.Engine,
            counts_tt.LayoutType,
            counts_tt.Engine,
            type_of(k_op),
            head_dim,
            rotary_dim,
            ratio,
            target,
        ](
            out_tt,
            gamma_tt,
            freqs_tt,
            counts_tt,
            k_op,
            Float32(atof(eps)),
            Int(key_counts.dim_size(0)),
            ctx,
        )
