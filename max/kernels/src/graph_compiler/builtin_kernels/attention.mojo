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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

"""Registers attention graph ops (MHA, MLA, fused QKV) and dispatches them to the `nn.attention` kernels."""

from std.collections import OptionalReg
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext
from layout.tile_tensor import row_major
from max.gpu.host.info import is_cpu, is_gpu
from kv_cache.paged_sparse_kv_index_remap import paged_sparse_kv_index_remap
from kv_cache.types import KVCacheStaticParams
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from nn.attention.cpu.mha import flash_attention as nn_flash_attention
from nn.attention.cpu.mha import flash_attention_split_kv
from nn.kv_cache import (
    generic_flash_attention_kv_cache_padded,
    generic_fused_qk_rope_bshd_paged,
    generic_fused_qkv_matmul_kv_cache_bshd_paged,
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
)
from nn.kv_cache_ragged import (
    generic_cross_attention_kv_cache,
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_decompress_k_cache_ragged_paged,
    generic_flare_mla_prefill_kv_cache_ragged,
    generic_flare_mla_prefill_ragged_paged_plan,
    generic_flash_attention_kv_cache_ragged,
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4,
)
from nn.attention.gpu.mha import flash_attention, flash_attention_ragged
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_utils import as_dynamic_row_major_1d, dispatch_mask
from nn.attention.gpu.mla_graph import (
    mla_prefill_branch_fp8,
    mla_prefill_branch_bf16,
    mla_decode_branch_fp8,
    mla_decode_branch_bf16,
    mla_prefill_decode_graph_fp8,
    mla_prefill_decode_graph_bf16,
)
from nn.attention.gpu.mla_index_fp8 import mla_indexer_ragged_float8_paged
from nn.attention.gpu.mla_decode_dispatch_scalars import (
    mla_decode_dispatch_scalars,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill import (
    mla_sm100_prefill_sparse,
    mla_sm100_prefill_sparse_fp8,
)
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from extensibility import DynamicTensor, InputTensor, OutputTensor
from extensibility import Tensor
from extensibility import (
    _FusedInputTensor as FusedInputTensor,
)
from extensibility import (
    _MutableInputTensor as MutableInputTensor,
)
from std.memory import UnsafePointer
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _execute_mha_ragged_paged_rel_logits,
    _execute_mha_ragged_paged_scalar_args,
    _unmarshal_mha_decode_dispatch_metadata,
    _unsafe_str_to_coord,
)


@extensibility.register("mo.mla.indexer.ragged.float8.paged")
struct MLAIndexerRaggedFloat8Paged:
    """Registers the `mo.mla.indexer.ragged.float8.paged` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        *,
        num_heads: Int,
        depth: Int,
        k: Int,
        quantization_granularity: Int,
        mask_str: StaticString,
        kpool: Int = 1,
    ](
        output_indices: OutputTensor[dtype=.int32, rank=2, ...],
        q: InputTensor[dtype=.float8_e4m3fn, rank=3, ...],
        qs: InputTensor[dtype=.float32, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        k_blocks: MutableInputTensor[dtype=.float8_e4m3fn, rank=6, ...],
        k_cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        k_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        k_max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        k_max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        k_scales: MutableInputTensor[dtype=.float32, rank=6, ...],
        # Resolves a request's scale pages. Pass `k_lookup_table` itself when
        # the scales share the values' block-id space; pass a distinct table
        # when they are paged independently.
        k_scales_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        """Compute FP8 attention scores and return top-k key indices per token.

        This kernel is designed for Multi-head Latent Attention (MLA) architectures.
        It computes FP8 matmul between queries and cached keys (with scales), applies
        masking, and returns the indices of the top-k highest-scoring keys per token.
        Scores are aggregated (summed) across all attention heads.

        Parameters:
            num_heads: Number of query attention heads (must be 128).
            depth: Head dimension (must be 128).
            k: Number of top indices to return per token.
            quantization_granularity: Quantization granularity for the K cache.
            mask_str: Mask type - either MaskName.NULL (no mask) or MaskName.CAUSAL.
            kpool: Tokens per pooled cache row. `1` scores one row per token;
                `k > 1` scores one pooled key per `kpool` consecutive tokens,
                so the K cache holds pooled keys and `k` counts pools.

        Args:
            output_indices: Output tensor [total_seq_len, top_k] containing
                top-k key indices per token. Invalid positions (where there are
                fewer than top_k valid keys) are filled with -1.
            q: Query tensor [total_seq_len, num_heads, depth] in FP8.
            qs: Query scales [total_seq_len, num_heads] in float32.
            input_row_offsets: Ragged row offsets [batch_size + 1] for queries.
            k_blocks: Paged K cache blocks [num_blocks, 1, num_layers, page_size,
                num_heads, head_size] in FP8.
            k_cache_lengths: Cache lengths [batch_size] - number of cached tokens
                per sequence.
            k_lookup_table: Page lookup table [batch_size, pages_per_seq] mapping
                sequence pages to block indices.
            k_max_prompt_length: Max prompt (query) length scalar tensor [1].
            k_max_cache_length: Max cache length scalar tensor [1].
            k_scales: K scale blocks matching k_blocks shape with scale values.
            k_scales_lookup_table: Page lookup table for the scale blocks
                [batch_size, pages_per_seq]. Equal to `k_lookup_table` when the
                scales share the values' block-id space.
            layer_idx: Layer index for retrieving the correct cache layer.
            ctx: Device context for GPU execution.
        """
        # Extract cache parameters from block shapes
        comptime page_size = Int(k_blocks.static_spec.shape_tuple[3])
        comptime head_dim = Int(k_blocks.static_spec.shape_tuple[5])
        comptime k_num_heads = Int(k_blocks.static_spec.shape_tuple[4])
        comptime is_mla = Int(k_blocks.static_spec.shape_tuple[1]) == 1
        comptime kv_params = KVCacheStaticParams(k_num_heads, head_dim, is_mla)
        comptime assert quantization_granularity >= depth, (
            "quantization_granularity must be >= depth for MLA (one scale per"
            " token per head)"
        )

        # K cache with scales (k_s values are stored in k_collection.scales)
        var k_collection = generic_get_paged_cache_with_scales[
            DType.float8_e4m3fn,
            DType.float32,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            LayoutTensor[.float8_e4m3fn, Layout.row_major[6](), MutAnyOrigin](
                k_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    k_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
                k_cache_lengths.to_layout_tensor().ptr,
                RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                    k_cache_lengths.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                k_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    k_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                k_max_prompt_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    k_max_prompt_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                k_max_cache_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    k_max_cache_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.float32, Layout.row_major[6](), MutAnyOrigin](
                k_scales.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    k_scales.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                k_scales_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    k_scales_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
        )

        mla_indexer_ragged_float8_paged[
            DType.float8_e4m3fn,
            type_of(k_collection),
            num_heads,
            depth,
            k,
            mask_str,
            kpool=kpool,
        ](
            output_indices.to_tile_tensor[.int64](),
            q.to_tile_tensor[.int64](),
            qs.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            k_collection,
            layer_idx,
            ctx,
        )


@extensibility.register("mo.composite.masked_flash_attention_gpu")
struct MaskedFlashAttentionGPU:
    """Registers the `mo.composite.masked_flash_attention_gpu` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString, rank: Int
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[rank=rank, ...],
        k: InputTensor[rank=rank, ...],
        v: InputTensor[rank=rank, ...],
        mask: InputTensor,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """`masked_flash_attention_gpu` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNorm = attentionMatrix * scale

        # Note attention_mask is HSS and auto-broadcasts
        attentionMatrixNormMasked = attentionMatrixNorm + attention_mask

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Compared to the CPU patterns the notable differences are:
        1. The mask is rank 3 and is of shape BSS
        2. The transposes are part of the kernel itself

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.

        Parameters:
            target: Target device identifier; must resolve to a GPU.
            rank: Rank of the `q`, `k`, `v`, and `output` tensors.

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in BSHD layout.
            k: Key tensor in BSHD layout, or BShD for grouped attention.
            v: Value tensor in BSHD layout, or BShD for grouped attention.
            mask: Attention mask tensor of rank 3 and shape BSS, added to
                the scaled attention scores before softmax.
            scale: Scaling factor applied to attention scores before softmax.
            ctx: Device context for GPU execution.
        """
        comptime assert is_gpu[target](), "only valid on GPUs"

        flash_attention(
            output.to_layout_tensor(),
            q.to_layout_tensor(),
            k.to_layout_tensor(),
            v.to_layout_tensor(),
            mask.to_layout_tensor(),
            scale,
            context=ctx,
        )


@extensibility.register("mo.mha.no_cache")
struct FlashAttentionGPU:
    """Registers the `mo.mha.no_cache` graph op with the graph compiler."""

    @staticmethod
    def execute[
        rank: Int,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[rank=rank, ...],
        k: InputTensor[rank=rank, ...],
        v: InputTensor[rank=rank, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """`mo.mha.no_cache` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNormMasked = mask_functor(attentionMatrix * scale)

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Compared to the CPU patterns the notable differences are:
        1. The transposes are part of the kernel itself

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.

        Parameters:
            rank: Rank of the `q`, `k`, `v`, and `output` tensors
                (inferred).
            target: Target device identifier; must resolve to a GPU.
            mask_str: Attention mask type string passed to
                `dispatch_mask` to select the mask functor.
            local_window_size: Sliding window size for windowed attention
                (defaults to -1, meaning no window).

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in BSHD layout.
            k: Key tensor in BSHD layout, or BShD for grouped attention.
            v: Value tensor in BSHD layout, or BShD for grouped attention.
            scale: Scaling factor applied to attention scores before softmax.
            ctx: Device context for GPU execution.
        """
        comptime assert is_gpu[target](), "only valid on GPUs"

        var output_buffer = output.to_layout_tensor()
        var q_buffer = q.to_layout_tensor()
        var k_buffer = k.to_layout_tensor()
        var v_buffer = v.to_layout_tensor()

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention[](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                scale,
                ctx,
            )

        dispatch_mask[mask_str, local_window_size](_dispatch_flash_attention)


@extensibility.register("mo.mha.padded.no_cache")
struct PaddedFlashAttentionGPU:
    """Registers the `mo.mha.padded.no_cache` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        rank: Int,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[rank=rank, ...],
        k: InputTensor[rank=rank, ...],
        v: InputTensor[rank=rank, ...],
        valid_length: InputTensor[dtype=.uint32, rank=1, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        comptime assert is_gpu[target](), "only valid on GPUs"

        var output_buffer = output.to_layout_tensor()
        var q_buffer = q.to_layout_tensor()
        var k_buffer = k.to_layout_tensor()
        var v_buffer = v.to_layout_tensor()

        comptime valid_length_t = LayoutTensor[
            .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
        var _valid_length = rebind[valid_length_t](
            valid_length.to_layout_tensor()
        )

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention[
                _use_valid_length=True,
                _padded_ndbuffer=True,
            ](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                scale,
                ctx,
                valid_length=OptionalReg[valid_length_t](_valid_length),
            )

        dispatch_mask[mask_str, local_window_size](_dispatch_flash_attention)


@extensibility.register("mo.mha.ragged.no_cache")
struct RaggedFlashAttentionGPU:
    """Registers the `mo.mha.ragged.no_cache` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        rank: Int,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank, ...],
        q: InputTensor[rank=rank, ...],
        k: InputTensor[rank=rank, ...],
        v: InputTensor[rank=rank, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        q_max_seq_len: InputTensor[dtype=.uint32, rank=1, ...],
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        """`mo.mha.ragged.no_cache` computes flash attention for ragged inputs without KV cache.

        The inputs q, k, v are in ragged format with shape [total_seq_len, num_heads, head_dim].
        input_row_offsets indicates where each sequence starts and ends in the ragged tensors.

        Parameters:
            rank: Rank of the `q`, `k`, `v`, and `output` tensors
                (inferred).
            target: Target device identifier; must resolve to a GPU.
            mask_str: Attention mask type string passed to
                `dispatch_mask` to select the mask functor.
            local_window_size: Sliding window size for windowed attention
                (defaults to -1, meaning no window).

        Args:
            output: Output attention tensor with the same shape as `q`.
            q: Query tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            k: Key tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            v: Value tensor in ragged format with shape
                [total_seq_len, num_heads, head_dim].
            input_row_offsets: Row offsets [batch_size + 1] marking the
                start and end of each sequence in the ragged tensors.
            q_max_seq_len: Maximum query sequence length across the batch.
            scale: Scaling factor applied to attention scores before softmax.
            ctx: Device context for GPU execution.
        """
        comptime assert is_gpu[target](), "only valid on GPUs"

        var output_buffer = output.to_layout_tensor()
        var q_buffer = q.to_layout_tensor()
        var k_buffer = k.to_layout_tensor()
        var v_buffer = v.to_layout_tensor()

        comptime input_row_offsets_t = LayoutTensor[
            .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
        var _input_row_offsets = rebind[input_row_offsets_t](
            input_row_offsets.to_layout_tensor()
        )

        def _dispatch_flash_attention[
            mask_t: MHAMask
        ](mask: mask_t) raises {
            var output_buffer,
            var q_buffer,
            var k_buffer,
            var v_buffer,
            imm,
        }:
            flash_attention_ragged[](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                _input_row_offsets,
                q_max_seq_len.to_layout_tensor(),
                mask,
                scale,
                ctx,
            )

        dispatch_mask[mask_str, local_window_size](_dispatch_flash_attention)


@extensibility.register("mo.composite.no_mask_flash_attention_cpu")
struct NoMaskFlashAttentionCPU:
    """Registers the `mo.composite.no_mask_flash_attention_cpu` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        q: InputTensor[dtype=dtype, rank=rank, ...],
        k: FusedInputTensor[dtype=dtype, rank=rank, ...],
        v: FusedInputTensor[dtype=dtype, rank=rank, ...],
        scale: Scalar[dtype=DType.float32],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_cpu[target](), "only valid on CPUs"

        @__parameter
        @always_inline
        def k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @__parameter
        @always_inline
        def v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @__parameter
        @always_inline
        def mask_input_fn[
            width: Int, _rank: Int
        ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
            return SIMD[dtype, width](0)

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            q.to_layout_tensor(),
            k.shape(),
            v.shape(),
            IndexList[0](),
            output.to_layout_tensor(),
            scale.cast[.float32](),
            ctx=Optional[DeviceContext](ctx),
        )


@extensibility.register("with_mask_flash_attention_split_kv_cpu")
struct WithMaskFlashAttentionSplitKVCPU:
    """Registers the `with_mask_flash_attention_split_kv_cpu` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        q: InputTensor[dtype=dtype, rank=rank, ...],
        k: FusedInputTensor[dtype=dtype, rank=rank, ...],
        v: FusedInputTensor[dtype=dtype, rank=rank, ...],
        k_cache: FusedInputTensor[dtype=dtype, rank=rank + 1, ...],
        v_cache: FusedInputTensor[dtype=dtype, rank=rank + 1, ...],
        mask: FusedInputTensor[dtype=dtype, ...],
        scale: Scalar[dtype=DType.float32],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_cpu[target](), "only valid on CPUs"

        @__parameter
        @always_inline
        def k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @__parameter
        @always_inline
        def v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @__parameter
        @always_inline
        def k_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k_cache.dtype, width]:
            return k_cache._lambda_load[width=width](
                rebind[IndexList[k_cache.rank]](coords)
            )

        @__parameter
        @always_inline
        def v_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v_cache.dtype, width]:
            return v_cache._lambda_load[width=width](
                rebind[IndexList[v_cache.rank]](coords)
            )

        @__parameter
        @always_inline
        def mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.dtype, width]:
            return mask._lambda_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        flash_attention_split_kv[
            k_input_fn,
            v_input_fn,
            k_cache_input_fn,
            v_cache_input_fn,
            mask_input_fn,
        ](
            q.to_layout_tensor(),
            k.shape(),
            v.shape(),
            k_cache.shape(),
            v_cache.shape(),
            mask.shape(),
            output.to_layout_tensor(),
            scale.cast[.float32](),
            ctx=Optional[DeviceContext](ctx),
        )


@extensibility.register_shape_function("with_mask_flash_attention_split_kv_cpu")
def with_mask_flash_attention_split_kv_cpu_shape(
    q: Some[Tensor],
    k: Some[Tensor],
    v: Some[Tensor],
    k_cache: Some[Tensor],
    v_cache: Some[Tensor],
    mask: Some[Tensor],
    scale: Scalar[dtype=DType.float32],
) -> IndexList[type_of(q).rank]:
    """Computes the output shape for the `with_mask_flash_attention_split_kv_cpu` graph op.
    """
    comptime assert (
        type_of(k).dtype == type_of(q).dtype
    ), "k and q must share a dtype"
    comptime assert (
        type_of(v).dtype == type_of(q).dtype
    ), "v and q must share a dtype"
    comptime assert (
        type_of(k_cache).dtype == type_of(q).dtype
    ), "k_cache and q must share a dtype"
    comptime assert (
        type_of(v_cache).dtype == type_of(q).dtype
    ), "v_cache and q must share a dtype"
    comptime assert (
        type_of(mask).dtype == type_of(q).dtype
    ), "mask and q must share a dtype"
    comptime assert (
        type_of(k).rank == type_of(q).rank
    ), "k and q must share a rank"
    comptime assert (
        type_of(v).rank == type_of(q).rank
    ), "v and q must share a rank"
    comptime assert (
        type_of(k_cache).rank == type_of(q).rank + 1
    ), "k_cache rank must be q rank + 1"
    comptime assert (
        type_of(v_cache).rank == type_of(q).rank + 1
    ), "v_cache rank must be q rank + 1"
    return rebind[IndexList[type_of(q).rank]](
        coord_to_index_list(q.shape().tuple())
    )


@extensibility.register("mo.composite.masked_flash_attention_cpu")
struct WithMaskFlashAttentionCPU:
    """Registers the `mo.composite.masked_flash_attention_cpu` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        q: InputTensor[dtype=dtype, rank=rank, ...],
        k: FusedInputTensor[dtype=dtype, rank=rank, ...],
        v: FusedInputTensor[dtype=dtype, rank=rank, ...],
        mask: FusedInputTensor[dtype=dtype, ...],
        scale: Scalar[dtype=DType.float32],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert is_cpu[target](), "only valid on CPUs"

        @__parameter
        @always_inline
        def k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @__parameter
        @always_inline
        def v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @__parameter
        @always_inline
        def mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.dtype, width]:
            return mask._lambda_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            q.to_layout_tensor(),
            k.shape(),
            v.shape(),
            mask.shape(),
            output.to_layout_tensor(),
            scale.cast[.float32](),
            ctx=Optional[DeviceContext](ctx),
        )


@extensibility.register("mo.fused_qkv_matmul.padded.paged")
struct Struct_fused_qkv_matmul_padded_paged:
    """Registers the `mo.fused_qkv_matmul.padded.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        hidden_state: InputTensor[dtype=dtype, rank=3, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        valid_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        var valid_lengths_lt = valid_lengths.to_layout_tensor()
        generic_fused_qkv_matmul_kv_cache_bshd_paged[target=target](
            hidden_state.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            LayoutTensor[
                .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ](
                valid_lengths_lt.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    valid_lengths_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged")
struct Struct_fused_qkv_matmul_padded_ragged:
    """Registers the `mo.fused_qkv_matmul.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_quantized:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.quantized` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        weight_type: DType,
        group_size: Int,
        has_zp_int: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=weight_type, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        comptime has_zp = True if has_zp_int == 1 else False
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.bias")
struct Struct_fused_qkv_matmul_padded_ragged_bias:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.bias` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.scale")
struct Struct_fused_qkv_matmul_padded_ragged_scale:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.scale` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            scales_granularity_mnk=IndexList[3](
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
            OptionalReg[
                LayoutTensor[
                    mut=False,
                    output_type,
                    Layout.row_major(UNKNOWN_VALUE),
                    ImmutAnyOrigin,
                    address_space=.GENERIC,
                ]
            ](),
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.scale.float4")
struct Struct_fused_qkv_matmul_padded_ragged_scale_float4:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.scale.float4` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=5, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=5, ...],
        tensor_sf: Float32,
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.scale.mxfp8")
struct Struct_fused_qkv_matmul_padded_ragged_scale_mxfp8:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.scale.mxfp8` graph op with the graph compiler.
    """

    # Delegates to the NVFP4 entry point, which is dual-mode and also handles
    # MXFP8 from its data dtype, scale dtype, and SF_VECTOR_SIZE parameters. The
    # "float4" in the callee name is intentional. Do not split off a separate
    # MXFP8 path.
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=5, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=5, ...],
        tensor_sf: Float32,
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.scale.mxfp8.amd")
struct Struct_fused_qkv_matmul_padded_ragged_scale_mxfp8_amd:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.scale.mxfp8.amd` graph op with the graph compiler.
    """

    # CDNA4 sibling of the struct above, delegating to the same dual-mode entry
    # point and the same band routing. It exists only because the scale layout
    # differs by vendor: SM100 wants the rank-5 SF-atom interleave, CDNA4
    # consumes the checkpoint's plain rank-2 [N, K // 32] E8M0 scales.
    # Everything below the entry point is shared.
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=2, ...],
        tensor_sf: Float32,
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale_float4[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            tensor_sf,
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8")
struct Struct_fused_qkv_index_matmul_padded_ragged_scale_mxfp8:
    """Registers the `mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8` graph op with the graph compiler.
    """

    # Dual-cache fused QKV + index-QK matmul for MiniMax-M3. Like the
    # single-cache mxfp8 struct above, this delegates to the dual-mode NVFP4
    # entry point, which also handles MXFP8 (E8M0 scales, SF_VECTOR_SIZE=32)
    # from its dtype/scale-dtype/SF_VECTOR_SIZE parameters. The "float4" in the
    # callee name is intentional; do not split off a separate MXFP8 path.
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q is returned in `q_output` [M, q_dim] and IndexQ
    # in `iq_output` [M, iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        IQ_DIM: Int,
        target: StaticString,
    ](
        q_output: OutputTensor[dtype=output_type, rank=2, ...],
        iq_output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=5, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=5, ...],
        tensor_sf: Float32,
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_blocks: MutableInputTensor[dtype=index_kv_type, rank=6, ...],
        index_cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        index_max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks,
            index_cache_lengths,
            index_kv_lookup_table,
            index_max_prompt_length,
            index_max_cache_length,
        )
        return (
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                target=target,
            ](
                hidden_state.to_layout_tensor(),
                input_row_offsets.to_layout_tensor(),
                weight.to_layout_tensor(),
                input_scale.to_layout_tensor(),
                weight_scale.to_layout_tensor(),
                tensor_sf,
                kv_collection,
                index_kv_collection,
                layer_idx,
                IQ_DIM,
                q_output.to_layout_tensor(),
                iq_output.to_layout_tensor(),
                ctx,
            )
        )


@extensibility.register(
    "mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8.amd"
)
struct Struct_fused_qkv_index_matmul_padded_ragged_scale_mxfp8_amd:
    """Registers the `mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8.amd` graph op with the graph compiler.
    """

    # CDNA4 sibling of the struct above, delegating to the same dual-mode
    # entry point and the same column routing. It exists only because the
    # scale layout differs by vendor: SM100 wants the rank-5 SF-atom
    # interleave, CDNA4 consumes the checkpoint's plain rank-2 [N, K // 32]
    # E8M0 scales. Everything below the entry point is shared.
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q is returned in `q_output` [M, q_dim] and IndexQ
    # in `iq_output` [M, iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        SF_VECTOR_SIZE: Int,
        IQ_DIM: Int,
        target: StaticString,
    ](
        q_output: OutputTensor[dtype=output_type, rank=2, ...],
        iq_output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=2, ...],
        tensor_sf: Float32,
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_blocks: MutableInputTensor[dtype=index_kv_type, rank=6, ...],
        index_cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        index_max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks,
            index_cache_lengths,
            index_kv_lookup_table,
            index_max_prompt_length,
            index_max_cache_length,
        )
        return (
            generic_fused_qkv_index_matmul_kv_cache_paged_ragged_scale_float4[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                target=target,
            ](
                hidden_state.to_layout_tensor(),
                input_row_offsets.to_layout_tensor(),
                weight.to_layout_tensor(),
                input_scale.to_layout_tensor(),
                weight_scale.to_layout_tensor(),
                tensor_sf,
                kv_collection,
                index_kv_collection,
                layer_idx,
                IQ_DIM,
                q_output.to_layout_tensor(),
                iq_output.to_layout_tensor(),
                ctx,
            )
        )


@extensibility.register("mo.fused_qkv_index_matmul.ragged.paged")
struct Struct_fused_qkv_index_matmul_padded_ragged:
    # BF16 (non-scaled) dual-cache fused QKV + index-QK matmul for MiniMax-M3.
    # Non-scaled analog of the mxfp8 struct above: same dual-cache column routing
    # (MAIN K/V scatter + INDEX IndexK scatter) and `IQ_DIM` parameter, but plain
    # BF16 with no block-scaling operands (no input_scale / weight_scale /
    # tensor_sf / SF_VECTOR_SIZE). Hardware-agnostic (AMD CDNA4 + NVIDIA).
    #
    # The MAIN cache operands (kv_blocks .. max_cache_length) drive the K/V
    # scatter; the INDEX cache operands (index_kv_blocks .. index_max_cache_length)
    # drive the IndexK scatter. Q and IndexQ are returned in the combined
    # `output` tensor [M, q_dim + iq_dim].
    #
    # `IQ_DIM` is the IndexQ output-band width (num_index_heads * idx_head_dim).
    # It is a parameter because, for the MLA index cache, it cannot be recovered
    # from the index cache's `num_heads` (== 1 for the single latent head).
    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        kv_type: DType,
        index_kv_type: DType,
        //,
        IQ_DIM: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_blocks: MutableInputTensor[dtype=index_kv_type, rank=6, ...],
        index_cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        index_kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        index_max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        index_max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        var index_kv_collection = generic_get_paged_cache(
            index_kv_blocks,
            index_cache_lengths,
            index_kv_lookup_table,
            index_max_prompt_length,
            index_max_cache_length,
        )
        return generic_fused_qkv_index_matmul_kv_cache_paged_ragged[
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            kv_collection,
            index_kv_collection,
            layer_idx,
            IQ_DIM,
            output.to_layout_tensor(),
            ctx,
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.scale.bias")
struct Struct_fused_qkv_matmul_padded_ragged_scale_bias:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.scale.bias` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=dtype, rank=2, ...],
        input_scale: InputTensor[dtype=scale_type, rank=2, ...],
        weight_scale: InputTensor[dtype=scale_type, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=kv_type, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        bias: InputTensor[dtype=output_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        comptime ExpectedBiasType = LayoutTensor[
            mut=False,
            output_type,
            Layout.row_major(UNKNOWN_VALUE),
            ImmutAnyOrigin,
            address_space=.GENERIC,
        ]
        var bias_tensor = bias.to_layout_tensor()
        var rebound_bias = rebind[ExpectedBiasType](bias_tensor)
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            scales_granularity_mnk=IndexList[3](
                m_scale_granularity, n_scale_granularity, k_scale_granularity
            ),
            target=target,
        ](
            hidden_state.to_layout_tensor(),
            input_row_offsets.to_layout_tensor(),
            weight.to_layout_tensor(),
            input_scale.to_layout_tensor(),
            weight_scale.to_layout_tensor(),
            kv_collection,
            layer_idx,
            output.to_layout_tensor(),
            ctx,
            OptionalReg[ExpectedBiasType](rebound_bias),
        )


@extensibility.register("mo.fused_qkv_matmul.ragged.paged.bias.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_bias_quantized:
    """Registers the `mo.fused_qkv_matmul.ragged.paged.bias.quantized` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        weight_type: DType,
        group_size: Int,
        has_zp_int: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2, ...],
        hidden_state: InputTensor[dtype=dtype, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        weight: InputTensor[dtype=weight_type, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        bias: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        comptime has_zp = True if has_zp_int == 1 else False
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


@extensibility.register("mo.fused_qk_rope.ragged.paged.with_position_id")
struct Struct_fused_qk_rope_ragged_paged_with_position_id[interleaved: Bool]:
    """Registers the `mo.fused_qk_rope.ragged.paged.with_position_id` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        mrope_section: StaticString,
        target: StaticString,
        cache_dtype: DType = dtype,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q_proj: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        position_ids: InputTensor[dtype=.uint32, rank=2, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        comptime mrope = _unsafe_str_to_coord[mrope_section]()
        generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
            interleaved=Self.interleaved,
            has_position_ids=True,
            target=target,
            mrope_types=mrope.element_types,
            mrope_section=mrope,
        ](
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            position_ids,
            layer_idx,
            output,
            context,
        )


@extensibility.register("mo.fused_qk_rope.ragged.paged")
struct Struct_fused_qk_rope_ragged_paged[interleaved: Bool]:
    """Registers the `mo.fused_qk_rope.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        freq_dtype: DType,
        //,
        target: StaticString,
        cache_dtype: DType = dtype,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q_proj: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        # Dummy position_ids - won't be used since has_position_ids=False
        var dummy_position_ids = DynamicTensor[dtype=DType.uint32, rank=2, ...](
            UnsafePointer[UInt32, MutAnyOrigin].unsafe_dangling(),
            IndexList[2](0),
        )
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
            interleaved=Self.interleaved,
            has_position_ids=False,
            target=target,
        ](
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            dummy_position_ids,
            layer_idx,
            output,
            context,
        )


@extensibility.register("mo.fused_qk_rope.padded.paged")
struct Struct_fused_qk_rope_padded_paged[interleaved: Bool]:
    """Registers the `mo.fused_qk_rope.padded.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        q_proj: InputTensor[dtype=dtype, rank=4, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=dtype, rank=2, ...],
        layer_idx: UInt32,
        valid_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_fused_qk_rope_bshd_paged[
            interleaved=Self.interleaved,
            target=target,
        ](
            q_proj.to_tile_tensor[.int64](),
            kv_collection,
            freqs_cis.to_tile_tensor[.int64](),
            layer_idx,
            valid_lengths.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.mha.padded.paged")
struct Struct_mha_padded_paged:
    """Registers the `mo.mha.padded.paged` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        q: InputTensor[dtype=dtype, rank=4, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        valid_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        scale: Float32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        var valid_lengths_lt = valid_lengths.to_layout_tensor()
        generic_flash_attention_kv_cache_padded[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
        ](
            q.to_layout_tensor(),
            kv_collection,
            layer_idx,
            LayoutTensor[
                .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
            ](
                valid_lengths_lt.ptr,
                RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                    valid_lengths_lt.runtime_layout.shape.value.canonicalize()
                ),
            ),
            scale,
            output.to_layout_tensor(),
            context,
        )


@extensibility.register("mo.mha.decode.get_num_partitions")
struct Struct_mha_decode_num_partitions:
    """Registers the `mo.mha.decode.get_num_partitions` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        *, n_kv_heads: Int
    ](
        num_partitions: OutputTensor[dtype=.int64, rank=1, ...],
        decode_num_partitions_request: InputTensor[
            dtype=DType.int64, rank=1, ...
        ],
        context: DeviceContext,
    ) raises:
        if decode_num_partitions_request.dim_size[0]() != 2:
            raise Error(
                "Expected decode_num_partitions_request to have shape [2]."
            )

        var request_ptr = decode_num_partitions_request.unsafe_ptr()
        var batch_size = Int(request_ptr[0])
        var max_cache_valid_length = Int(request_ptr[1])

        if batch_size < 1:
            raise Error(
                "decode_num_partitions_request[0] (batch size) must be "
                "positive."
            )

        if max_cache_valid_length < 0:
            raise Error(
                "decode_num_partitions_request[1] (max cache length) must be "
                "non-negative."
            )

        num_partitions[0] = Int64(
            mha_decoding_num_partitions(
                batch_size,
                max_cache_valid_length,
                n_kv_heads,
                context,
            )
        )


@extensibility.register("mo.mha.ragged.paged")
struct Struct_mha_ragged_paged_scalar_args:
    """Registers the `mo.mha.ragged.paged` graph op with the graph compiler."""

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        q_dtype: DType,
        cache_dtype: DType,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=out_dtype, rank=3, ...],
        q: InputTensor[dtype=q_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        mha_decode_dispatch_metadata: InputTensor[
            dtype=DType.int64, rank=1, ...
        ],
        context: DeviceContext,
    ) raises:
        _execute_mha_ragged_paged_scalar_args[
            target=target,
            mask_str=mask_str,
            local_window_size=local_window_size,
            output_dtype=out_dtype,
            cache_dtype=cache_dtype,
        ](
            output,
            q,
            input_row_offsets,
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
            layer_idx,
            scale,
            mha_decode_dispatch_metadata,
            context,
        )


@extensibility.register("mo.mha.ragged.paged.sink_weights")
struct Struct_mha_ragged_paged_sink_weights_scalar_args:
    """Registers the `mo.mha.ragged.paged.sink_weights` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
        mask_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        sink_weights: InputTensor[dtype=dtype, rank=1, ...],
        mha_decode_dispatch_metadata: InputTensor[
            dtype=DType.int64, rank=1, ...
        ],
        context: DeviceContext,
    ) raises:
        var sink_weights_lt = sink_weights.to_layout_tensor()
        var sink_weights_rebound = as_dynamic_row_major_1d(sink_weights_lt)
        _execute_mha_ragged_paged_scalar_args[
            target=target,
            mask_str=mask_str,
            sink=True,
            local_window_size=local_window_size,
            output_dtype=dtype,
            cache_dtype=dtype,
        ](
            output,
            q,
            input_row_offsets,
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
            layer_idx,
            scale,
            mha_decode_dispatch_metadata,
            context,
            OptionalReg[
                LayoutTensor[
                    dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
                ]
            ](sink_weights_rebound),
        )


@extensibility.register("mo.mha.ragged.paged.rel_logits")
struct Struct_mha_ragged_paged_rel_logits:
    """Registers the `mo.mha.ragged.paged.rel_logits` graph op.

    Adds a relative-position bias to pre-softmax scores, gathered by
    `rel_dist = q_pos - k_pos` via `RelativeLogitsMask` rather than
    materializing a dense `(q, k)` mask.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        q_dtype: DType,
        cache_dtype: DType,
        //,
        target: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=out_dtype, rank=3, ...],
        q: InputTensor[dtype=q_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        bias: InputTensor[dtype=q_dtype, rank=3, ...],
        mha_decode_dispatch_metadata: InputTensor[
            dtype=DType.int64, rank=1, ...
        ],
        context: DeviceContext,
    ) raises:
        _execute_mha_ragged_paged_rel_logits[
            target=target,
            local_window_size=local_window_size,
            output_dtype=out_dtype,
            cache_dtype=cache_dtype,
        ](
            output,
            q,
            input_row_offsets,
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
            layer_idx,
            scale,
            bias,
            mha_decode_dispatch_metadata,
            context,
        )


@extensibility.register("mo.mla.decode.ragged.paged")
struct Struct_mla_decode_ragged_paged:
    """Registers the `mo.mla.decode.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        q_dtype: DType,
        kv_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=q_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert (
            Int(kv_blocks.static_spec.shape_tuple[1]) == 1
        ), "Only support only_k=True for MLA decompress"
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor[.int64](),
            scalar_args.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.mla.decode.ragged.paged.scaled")
struct Struct_mla_decode_ragged_paged_scaled:
    """Registers the `mo.mla.decode.ragged.paged.scaled` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        q_dtype: DType,
        kv_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        per_token_scale_rope_aware: Int = 0,
        quantization_granularity: Int = 640,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=q_dtype, rank=3, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        kv_scales: MutableInputTensor[dtype=.float32, rank=6, ...],
        # Resolves a request's scale pages. Pass `kv_lookup_table` itself when
        # the scales share the values' block-id space; pass a distinct table
        # when they are paged independently.
        kv_scales_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        q_scales: InputTensor[dtype=.float32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert (
            Int(kv_blocks.static_spec.shape_tuple[1]) == 1
        ), "Only support only_k=True for MLA decompress"

        comptime page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        comptime head_dim = Int(kv_blocks.static_spec.shape_tuple[5])
        comptime kv_num_heads = Int(kv_blocks.static_spec.shape_tuple[4])
        comptime kv_params = KVCacheStaticParams(kv_num_heads, head_dim, True)

        var kv_collection = generic_get_paged_cache_with_scales[
            kv_dtype,
            DType.float32,
            kv_params,
            page_size,
            quantization_granularity,
        ](
            LayoutTensor[kv_dtype, Layout.row_major[6](), MutAnyOrigin](
                kv_blocks.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    kv_blocks.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout(UNKNOWN_VALUE), ImmutAnyOrigin](
                cache_lengths.to_layout_tensor().ptr,
                RuntimeLayout[Layout(UNKNOWN_VALUE)].row_major(
                    cache_lengths.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                kv_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    kv_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                max_prompt_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    max_prompt_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[1](), ImmutAnyOrigin](
                max_cache_length.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[1]()].row_major(
                    max_cache_length.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.float32, Layout.row_major[6](), MutAnyOrigin](
                kv_scales.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[6]()].row_major(
                    kv_scales.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
            LayoutTensor[.uint32, Layout.row_major[2](), ImmutAnyOrigin](
                kv_scales_lookup_table.to_layout_tensor().ptr,
                RuntimeLayout[Layout.row_major[2]()].row_major(
                    kv_scales_lookup_table.to_layout_tensor().runtime_layout.shape.value
                ),
            ),
        )

        # Get the q_scales raw pointer for per-token Q scaling.
        var q_scale_ptr = q_scales.to_layout_tensor().ptr

        generic_flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            per_token_scale_rope_aware=per_token_scale_rope_aware != 0,
        ](
            q.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor[.int64](),
            scalar_args.to_tile_tensor[.int64](),
            context,
            q_scale_ptr,
        )


@extensibility.register("mo.mla.prefill.ragged.paged")
struct Struct_mla_prefill_ragged_paged:
    """Registers the `mo.mla.prefill.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        qkv_dtype: DType,
        //,
        target: StaticString,
        mask_str: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=qkv_dtype, rank=3, ...],
        k: InputTensor[dtype=qkv_dtype, rank=3, ...],
        v: InputTensor[dtype=qkv_dtype, rank=3, ...],
        buffer_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=qkv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_flare_mla_prefill_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
        ](
            q.to_tile_tensor[.int64](),
            k.to_tile_tensor[.int64](),
            v.to_tile_tensor[.int64](),
            buffer_row_offsets.to_tile_tensor[.int64](),
            cache_offsets.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            scale,
            output.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.mla.prefill.ragged.plan")
struct Struct_mla_prefill_ragged_plan:
    """Registers the `mo.mla.prefill.ragged.plan` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        buffer_row_offsets: OutputTensor[dtype=.uint32, rank=2, ...],
        cache_offsets: OutputTensor[dtype=.uint32, rank=2, ...],
        buffer_lengths: OutputTensor[dtype=.int32, rank=1, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        buffer_tok_size: UInt32,
        context: DeviceContext,
    ) raises:
        comptime assert Int(kv_blocks.static_spec.shape_tuple[1]) == 1, (
            "Expected is_mla=True for MLA decompress, but found both k and"
            " v dimensions."
        )
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_flare_mla_prefill_ragged_paged_plan[target=target](
            input_row_offsets.to_layout_tensor(),
            kv_collection,
            layer_idx,
            buffer_tok_size,
            buffer_row_offsets.to_layout_tensor(),
            cache_offsets.to_layout_tensor(),
            buffer_lengths.to_layout_tensor(),
            context,
        )


@extensibility.register("mo.mla.decompress.k.cache.ragged.paged")
struct Struct_mla_decompress_k_cache_ragged_paged:
    """Registers the `mo.mla.decompress.k.cache.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        dtype: DType,
        //,
        target: StaticString,
    ](
        k_latent_buffer: OutputTensor[dtype=dtype, rank=2, ...],
        k_buffer: OutputTensor[dtype=dtype, rank=2, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        weight: InputTensor[dtype=dtype, rank=2, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_flare_mla_decompress_k_cache_ragged_paged[target=target](
            buffer_row_offsets_1d.to_layout_tensor(),
            cache_offsets_1d.to_layout_tensor(),
            buffer_length,
            weight.to_layout_tensor(),
            kv_collection,
            layer_idx,
            k_latent_buffer.to_layout_tensor(),
            k_buffer.to_layout_tensor(),
            context,
        )


@extensibility.register("mo.mla.graph.prefill.paged.fp8")
struct Struct_mla_prefill_graph_paged:
    """Registers the `mo.mla.graph.prefill.paged.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=fp8_dtype, rank=2, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: InputTensor[dtype=fp8_scale_dtype, rank=2, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.prefill.paged.fp8 is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        mla_prefill_branch_fp8[
            m_scale_granularity=m_scale_granularity,
            n_scale_granularity=n_scale_granularity,
            k_scale_granularity=k_scale_granularity,
            mask_str=mask_str,
            kv_input_fn=kv_input_fn,
            target=target,
        ](
            output.to_tile_tensor[.int64](),
            q.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            freqs_cis.to_tile_tensor[.int64](),
            kv_norm_gamma.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets_1d.to_tile_tensor[.int64](),
            cache_offsets_1d.to_tile_tensor[.int64](),
            Int(buffer_length),
            w_k.to_tile_tensor[.int64](),
            w_k_scale.to_tile_tensor[.int64](),
            w_uv.to_tile_tensor[.int64](),
            w_uv_scale.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.mla.compute_dispatch_args.scalar")
struct Struct_mla_compute_dispatch_args_scalar:
    """Registers the `mo.mla.compute_dispatch_args.scalar` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        num_heads: Int,
        is_fp8_kv: Bool,
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.int64, rank=1, ...],
        batch_size_tensor: InputTensor[dtype=DType.int64, rank=1, ...],
        max_cache_valid_length_tensor: InputTensor[
            dtype=DType.int64, rank=1, ...
        ],
        q_max_seq_len_tensor: InputTensor[dtype=DType.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.mla.compute_dispatch_args.scalar is only supported on GPU"

        var ctx = context
        var batch_size = Int(batch_size_tensor.unsafe_ptr()[0])
        var max_cache_valid_length = Int(
            max_cache_valid_length_tensor.unsafe_ptr()[0]
        )
        var q_max_seq_len = Int(q_max_seq_len_tensor.unsafe_ptr()[0])

        if batch_size < 0:
            raise Error("batch_size must be non-negative.")
        if batch_size == 0:
            output[0] = Int64(0)
            output[1] = Int64(q_max_seq_len)
            output[2] = Int64(1)
            return

        # Route through the device-generic helper so this op (the test
        # reference) stays in lockstep with the `mla_dispatch_args_scalar`
        # binding the runtime resolver calls: HIP -> AMD heuristic, CUDA ->
        # SM100 runtime heuristic.
        var scalars = mla_decode_dispatch_scalars(
            batch_size,
            max_cache_valid_length,
            q_max_seq_len,
            num_heads,
            is_fp8_kv,
            ctx,
        )

        output[0] = Int64(scalars[0])
        output[1] = Int64(scalars[1])
        output[2] = Int64(scalars[2])


@extensibility.register("mo.mla.graph.decode.paged.fp8")
struct Struct_mla_decode_graph_paged_fp8:
    """Registers the `mo.mla.graph.decode.paged.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=fp8_dtype, rank=3, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_uk_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.decode.paged.fp8 is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.decode.paged.fp8",
            task_id=get_safe_task_id(context),
        ):
            mla_decode_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uk_scale.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                w_uv_scale.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("mo.mla.graph.decode.paged.fp8.sparse")
struct Struct_mla_decode_graph_paged_fp8_sparse:
    """Registers the `mo.mla.graph.decode.paged.fp8.sparse` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        cache_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
        indices_stride: Int,
        # Read-once shared-index MTP fold (KERN-3141). Set from the Python
        # `index_share` op attribute (default False -> per-position path,
        # byte-identical). Forwarded to the decode dispatch as
        # `fold_shared_index`.
        index_share: Bool = False,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=fp8_dtype, rank=3, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_uk_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.decode.paged.fp8.sparse is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var topk_lengths_ptr = topk_lengths.to_layout_tensor().ptr
        var attn_sink_ptr = attn_sink.to_layout_tensor().ptr
        # `sparse_indices` still holds the logical key positions here; the
        # remap below produces the physical gather buffer and drops them.
        # Capture them for position-based causal masking (see
        # mla_decode_utils.mojo).
        var logical_indices_ptr = sparse_indices.to_layout_tensor().ptr

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.decode.paged.fp8.sparse",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, cache_dtype
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )
            var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])
            mla_decode_branch_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
                sparse_mla=True,
                fold_shared_index=index_share,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uk_scale.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                w_uv_scale.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                scratch_sparse_indices.unsafe_ptr().unsafe_origin_cast[
                    MutAnyOrigin
                ](),
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                # Sparse path: kernel caps effective_split_len via topk mask;
                # passing num_partitions would override that. Let the kernel
                # compute its own values.
                num_partitions_in=None,
                logical_indices=logical_indices_ptr,
            )


@extensibility.register("mo.mla.graph.prefill.paged")
struct Struct_mla_prefill_graph_bf16_paged:
    """Registers the `mo.mla.graph.prefill.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=.bfloat16, rank=2, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.prefill.paged is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        mla_prefill_branch_bf16[
            mask_str=mask_str,
            kv_input_fn=kv_input_fn,
            target=target,
        ](
            output.to_tile_tensor[.int64](),
            q.to_tile_tensor[.int64](),
            input_row_offsets.to_tile_tensor[.int64](),
            freqs_cis.to_tile_tensor[.int64](),
            kv_norm_gamma.to_tile_tensor[.int64](),
            kv_collection,
            layer_idx,
            scale,
            epsilon,
            buffer_row_offsets_1d.to_tile_tensor[.int64](),
            cache_offsets_1d.to_tile_tensor[.int64](),
            Int(buffer_length),
            w_k.to_tile_tensor[.int64](),
            w_uv.to_tile_tensor[.int64](),
            context,
        )


@extensibility.register("mo.mla.graph.decode.paged")
struct Struct_mla_decode_graph_bf16_paged:
    """Registers the `mo.mla.graph.decode.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.decode.paged is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.decode.paged",
            task_id=get_safe_task_id(context),
        ):
            mla_decode_branch_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("mo.mla.graph.decode.paged.sparse")
struct Struct_mla_decode_graph_bf16_paged_sparse:
    """Registers the `mo.mla.graph.decode.paged.sparse` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        indices_stride: Int,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.decode.paged.sparse is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var topk_lengths_ptr = topk_lengths.to_layout_tensor().ptr
        var attn_sink_ptr = attn_sink.to_layout_tensor().ptr

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.decode.paged.sparse",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, kv_dtype
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )
            mla_decode_branch_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
                sparse_mla=True,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                scratch_sparse_indices.unsafe_ptr().unsafe_origin_cast[
                    MutAnyOrigin
                ](),
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                num_partitions_in=None,
            )


@extensibility.register("mo.mla.graph.prefill.decode.paged.fp8")
struct Struct_mla_prefill_graph_decode_paged_fp8:
    """Registers the `mo.mla.graph.prefill.decode.paged.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        cache_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=fp8_dtype, rank=2, ...],
        w_uk: InputTensor[dtype=fp8_dtype, rank=3, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: InputTensor[dtype=fp8_scale_dtype, rank=2, ...],
        w_uk_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.prefill.decode.paged.fp8 is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.prefill.decode.paged.fp8",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.int64](),
                cache_offsets_1d.to_tile_tensor[.int64](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.int64](),
                w_k_scale.to_tile_tensor[.int64](),
                w_uk.to_tile_tensor[.int64](),
                w_uk_scale.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                w_uv_scale.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("mo.mla.graph.prefill.decode.paged.fp8.sparse")
struct Struct_mla_prefill_graph_decode_paged_fp8_sparse:
    """Registers the `mo.mla.graph.prefill.decode.paged.fp8.sparse` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        fp8_dtype: DType,
        fp8_scale_dtype: DType,
        cache_dtype: DType,
        //,
        m_scale_granularity: Int,
        n_scale_granularity: Int,
        k_scale_granularity: Int,
        mask_str: StaticString,
        target: StaticString,
        indices_stride: Int,
        # Read-once shared-index MTP fold (KERN-3141). Set from the Python
        # `index_share` op attribute (default False -> per-position path,
        # byte-identical). Forwarded to the decode dispatch as
        # `fold_shared_index`.
        index_share: Bool = False,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=fp8_dtype, rank=2, ...],
        w_uk: InputTensor[dtype=fp8_dtype, rank=3, ...],
        w_uv: InputTensor[dtype=fp8_dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        w_k_scale: InputTensor[dtype=fp8_scale_dtype, rank=2, ...],
        w_uk_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        w_uv_scale: InputTensor[dtype=fp8_scale_dtype, rank=3, ...],
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[target](), (
            "mo.mla.graph.prefill.decode.paged.fp8.sparse is only supported"
            " on GPU"
        )

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var topk_lengths_ptr = topk_lengths.to_layout_tensor().ptr
        var attn_sink_ptr = attn_sink.to_layout_tensor().ptr

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.prefill.decode.paged.fp8.sparse",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, cache_dtype
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )
            var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])
            mla_prefill_decode_graph_fp8[
                m_scale_granularity=m_scale_granularity,
                n_scale_granularity=n_scale_granularity,
                k_scale_granularity=k_scale_granularity,
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
                sparse_mla=True,
                sparse_indices_stride=indices_stride,
                fold_shared_index=index_share,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.int64](),
                cache_offsets_1d.to_tile_tensor[.int64](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.int64](),
                w_k_scale.to_tile_tensor[.int64](),
                w_uk.to_tile_tensor[.int64](),
                w_uk_scale.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                w_uv_scale.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                scratch_sparse_indices.unsafe_ptr().unsafe_origin_cast[
                    MutAnyOrigin
                ](),
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                # Sparse path: let kernel use its own mask-aware computation.
                num_partitions_in=None,
            )


@extensibility.register("mo.mla.prefill.sparse.paged")
struct Struct_mla_prefill_sparse_paged:
    """Registers the `mo.mla.prefill.sparse.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        cache_dtype: DType,
        //,
        target: StaticString,
        indices_stride: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.mla.prefill.sparse.paged is only supported on GPU"

        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        # The underlying kernel asserts qk_depth == 576, num_q_heads == 128,
        # num_kv_heads == 1; pull those from the input static shapes.
        comptime num_q_heads = Int(q.static_spec.shape_tuple[1])
        comptime qk_depth = Int(q.static_spec.shape_tuple[2])
        comptime v_depth = Int(output.static_spec.shape_tuple[2])
        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])

        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var attn_sink_ptr = UnsafePointer[Float32, origin=ImmutAnyOrigin](
            attn_sink.to_layout_tensor().ptr
        )

        var k_cache = kv_collection.get_key_cache(Int(layer_idx))

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.prefill.sparse.paged",
            task_id=get_safe_task_id(context),
        ):
            # Logical → physical sparse index remap. The kernel expects
            # each selected key as `Int32(physical_block_id * page_size +
            # token_offset_within_page)`; the indexer emits logical
            # `[0, cache_length)` positions, so we remap into a scratch
            # buffer here.
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, cache_dtype
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )

            # The kernel's `indices` / `topk_lengths` tile tensors are
            # `DType.uint32`. Invalid sparse slots are encoded as `Int32(-1)`
            # which has the same bit pattern as `UInt32(0xFFFFFFFF)`; the
            # producer in the kernel reads them as int32 and rejects the
            # negative ones (cf. the `idx >= 0` check in the gather4 path),
            # so reinterpreting the bits via `bitcast` is sound.
            var indices_tt = TileTensor(
                scratch_sparse_indices.unsafe_ptr().bitcast[UInt32](),
                row_major(num_indices_sparse),
            )
            var topk_lengths_tt = TileTensor(
                topk_lengths.to_layout_tensor().ptr.bitcast[UInt32](),
                row_major(Int(topk_lengths.dim_size(0))),
            )

            mla_sm100_prefill_sparse[
                num_q_heads=num_q_heads,
                qk_depth=qk_depth,
                v_depth=v_depth,
                indices_stride=indices_stride,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_ptr,
                scale,
                context,
            )


@extensibility.register("mo.mla.prefill.sparse.paged.fp8")
struct Struct_mla_prefill_sparse_paged_fp8:
    """Registers the `mo.mla.prefill.sparse.paged.fp8` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        dtype: DType,
        //,
        target: StaticString,
        indices_stride: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=3, ...],
        q: InputTensor[dtype=dtype, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=.float8_e4m3fn, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        kv_scales: InputTensor[dtype=.float32, rank=1, ...],
        scale: Float32,
        context: DeviceContext,
    ) raises:
        comptime assert is_gpu[
            target
        ](), "mo.mla.prefill.sparse.paged.fp8 is only supported on GPU"

        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime num_q_heads = Int(q.static_spec.shape_tuple[1])
        comptime qk_depth = Int(q.static_spec.shape_tuple[2])
        comptime v_depth = Int(output.static_spec.shape_tuple[2])
        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])

        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var attn_sink_ptr = UnsafePointer[Float32, origin=ImmutAnyOrigin](
            attn_sink.to_layout_tensor().ptr
        )

        var scales_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
            kv_scales.to_layout_tensor().ptr
        )

        var k_cache = kv_collection.get_key_cache(Int(layer_idx))

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.prefill.sparse.paged.fp8",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, DType.float8_e4m3fn
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )

            var indices_tt = TileTensor(
                scratch_sparse_indices.unsafe_ptr().bitcast[UInt32](),
                row_major(num_indices_sparse),
            )
            var topk_lengths_tt = TileTensor(
                topk_lengths.to_layout_tensor().ptr.bitcast[UInt32](),
                row_major(Int(topk_lengths.dim_size(0))),
            )

            mla_sm100_prefill_sparse_fp8[
                num_q_heads=num_q_heads,
                qk_depth=qk_depth,
                v_depth=v_depth,
                indices_stride=indices_stride,
                scale_block_size=qk_depth,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                k_cache,
                indices_tt,
                topk_lengths_tt,
                attn_sink_ptr,
                scales_ptr,
                scale,
                context,
            )


@extensibility.register("mo.mla.graph.prefill.decode.paged")
struct Struct_mla_prefill_graph_decode_bf16_paged:
    """Registers the `mo.mla.graph.prefill.decode.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=.bfloat16, rank=2, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.prefill.decode.paged is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.prefill.decode.paged",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.int64](),
                cache_offsets_1d.to_tile_tensor[.int64](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.int64](),
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("mo.mla.graph.prefill.decode.paged.sparse")
struct Struct_mla_prefill_graph_decode_paged_sparse:
    @always_inline
    @staticmethod
    @__parameter
    def execute[
        freq_dtype: DType,
        gamma_dtype: DType,
        cache_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        indices_stride: Int,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=.bfloat16, rank=2, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        sparse_indices: InputTensor[dtype=.int32, rank=2, ...],
        topk_lengths: InputTensor[dtype=.int32, rank=1, ...],
        attn_sink: InputTensor[dtype=.float32, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[
            target
        ](), "mo.mla.graph.prefill.decode.paged.sparse is only supported on GPU"

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        comptime mla_page_size = Int(kv_blocks.static_spec.shape_tuple[3])
        var dev_ctx = context
        var num_indices_sparse = sparse_indices.size()

        var topk_lengths_ptr = topk_lengths.to_layout_tensor().ptr
        var attn_sink_ptr = attn_sink.to_layout_tensor().ptr

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.prefill.decode.paged.sparse",
            task_id=get_safe_task_id(context),
        ):
            var scratch_sparse_indices = dev_ctx.enqueue_create_buffer[
                DType.int32
            ](num_indices_sparse)
            paged_sparse_kv_index_remap[
                target, mla_page_size, indices_stride, cache_dtype
            ](
                scratch_sparse_indices.unsafe_ptr(),
                sparse_indices,
                input_row_offsets,
                kv_lookup_table,
                kv_blocks,
                context,
            )
            var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
                sparse_mla=True,
                sparse_indices_stride=indices_stride,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.int64](),
                cache_offsets_1d.to_tile_tensor[.int64](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.int64](),
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                scratch_sparse_indices.unsafe_ptr().unsafe_origin_cast[
                    MutAnyOrigin
                ](),
                indices_stride,
                topk_lengths_ptr,
                attn_sink_ptr,
                # Sparse path: let kernel use its own mask-aware computation.
                num_partitions_in=None,
            )


@extensibility.register("mo.mla.graph.prefill.decode.paged.quantized")
struct Struct_mla_prefill_graph_decode_bf16_paged_quantized:
    """Registers the `mo.mla.graph.prefill.decode.paged.quantized` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    @__parameter
    def execute[
        kv_dtype: DType,
        freq_dtype: DType,
        gamma_dtype: DType,
        scales_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=.bfloat16, rank=3, ...],
        q: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv: FusedInputTensor[dtype=.bfloat16, rank=2, ...],
        input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        freqs_cis: InputTensor[dtype=freq_dtype, rank=2, ...],
        kv_norm_gamma: InputTensor[dtype=gamma_dtype, rank=1, ...],
        buffer_row_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        cache_offsets_1d: InputTensor[dtype=.uint32, rank=1, ...],
        buffer_length: Int32,
        w_k: InputTensor[dtype=.bfloat16, rank=2, ...],
        w_uk: InputTensor[dtype=.bfloat16, rank=3, ...],
        w_uv: InputTensor[dtype=.bfloat16, rank=3, ...],
        kv_blocks: MutableInputTensor[dtype=kv_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        kv_scales: MutableInputTensor[dtype=scales_dtype, rank=6, ...],
        layer_idx: UInt32,
        scale: Float32,
        epsilon: Float32,
        scalar_args: InputTensor[dtype=.int64, rank=1, ...],
        num_partitions_scalar: InputTensor[dtype=.int64, rank=1, ...],
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )

        comptime assert is_gpu[target](), (
            "mo.mla.graph.prefill.decode.paged.quantized is only supported"
            " on GPU"
        )

        @__parameter
        @always_inline
        def kv_input_fn[
            width: Int
        ](coords: IndexList[2]) -> SIMD[.bfloat16, width]:
            return kv._lambda_load[width=width, element_alignment=width](coords)

        var num_partitions_proj = Int(num_partitions_scalar.unsafe_ptr()[0])

        with Trace[TraceLevel.OP, target=target](
            "mo.mla.graph.prefill.decode.paged.quantized",
            task_id=get_safe_task_id(context),
        ):
            mla_prefill_decode_graph_bf16[
                mask_str=mask_str,
                kv_input_fn=kv_input_fn,
                target=target,
            ](
                output.to_tile_tensor[.int64](),
                q.to_tile_tensor[.int64](),
                input_row_offsets.to_tile_tensor[.int64](),
                freqs_cis.to_tile_tensor[.int64](),
                kv_norm_gamma.to_tile_tensor[.int64](),
                kv_collection,
                layer_idx,
                scale,
                epsilon,
                buffer_row_offsets_1d.to_tile_tensor[.int64](),
                cache_offsets_1d.to_tile_tensor[.int64](),
                Int(buffer_length),
                Int(kv_collection.max_seq_length),
                w_k.to_tile_tensor[.int64](),
                w_uk.to_tile_tensor[.int64](),
                w_uv.to_tile_tensor[.int64](),
                scalar_args.to_tile_tensor[.int64](),
                context,
                num_partitions_in=num_partitions_proj,
            )


@extensibility.register("mo.cross_attention.ragged.paged")
struct Struct_cross_attention_ragged_paged:
    """Registers the `mo.cross_attention.ragged.paged` graph op with the graph compiler.
    """

    @always_inline
    @staticmethod
    def execute[
        out_dtype: DType,
        q_dtype: DType,
        cache_dtype: DType,
        //,
        mask_str: StaticString,
        target: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=out_dtype, rank=3, ...],
        q: InputTensor[dtype=q_dtype, rank=3, ...],
        q_input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        q_max_seq_len: InputTensor[dtype=.uint32, rank=1, ...],
        kv_input_row_offsets: InputTensor[dtype=.uint32, rank=1, ...],
        kv_blocks: MutableInputTensor[dtype=cache_dtype, rank=6, ...],
        cache_lengths: InputTensor[dtype=.uint32, rank=1, ...],
        kv_lookup_table: InputTensor[dtype=.uint32, rank=2, ...],
        max_prompt_length: InputTensor[dtype=.uint32, rank=1, ...],
        max_cache_length: InputTensor[dtype=.uint32, rank=1, ...],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContext,
    ) raises:
        var kv_collection = generic_get_paged_cache(
            kv_blocks,
            cache_lengths,
            kv_lookup_table,
            max_prompt_length,
            max_cache_length,
        )
        generic_cross_attention_kv_cache[
            mask_str=mask_str,
            local_window_size=local_window_size,
            target=target,
            output_dtype=out_dtype,
        ](
            q.to_layout_tensor(),
            q_input_row_offsets.to_layout_tensor(),
            q_max_seq_len.to_layout_tensor(),
            kv_input_row_offsets.to_layout_tensor(),
            kv_collection,
            layer_idx,
            scale,
            output.to_layout_tensor(),
            context,
        )
