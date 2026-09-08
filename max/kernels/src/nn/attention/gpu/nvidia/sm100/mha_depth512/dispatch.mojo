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
"""Dispatch for depth=256/512 pair-CTA SM100 (Blackwell) MHA prefill.

Creates the Depth512SM100Config, TMA tile descriptors, and launches the
pair-CTA kernel with cluster_dim=(2,1,1). The TransientScheduler uses
pair_cta=True so that both CTAs in a cluster derive the same tile index
from block_idx.x >> 1.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from max.gpu.host import DeviceContext, Dim, FuncAttribute, DeviceBuffer
from layout import TensorEngine
from layout.tma_async import RaggedTMA3DTile
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.logger import Logger
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    o_store_tma_blocks_per_op,
)
from .config import Depth512SM100Config
from .kernel import SM100MHADepth512


comptime logger = Logger()


@always_inline
def mha_sm100_depth512_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    KVRowOffsetsEngine: TensorEngine,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type], _],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    valid_length: UnsafePointer[UInt32, _],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        ImmutTileTensor1D[.uint32, Engine=KVRowOffsetsEngine]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
) raises:
    """Dispatches the pair-CTA SM100 depth=256/512 MHA prefill kernel.

    Builds the `Depth512SM100Config`, constructs TMA tile descriptors for the
    Q, K, V, and output operands, creates a `TransientScheduler` with
    `pair_cta=True`, and enqueues the `SM100MHADepth512` kernel with
    `cluster_dim=(2, 1, 1)` so both CTAs in a cluster derive the same tile
    index from `block_idx.x >> 1`. Only prefill is supported; decoding is
    rejected at compile time.

    Parameters:
        q_type: The query tensor element type (inferred).
        KVType: The paged KV cache operand type providing the tile factory
            and page size (inferred).
        MaskType: The attention mask type applied to the scores (inferred).
        output_type: The output buffer element type (inferred).
        MaxPromptLenType: The maximum prompt length as a static or runtime
            value (inferred).
        PartitionType: The KV cache partition scheme (inferred).
        KVRowOffsetsEngine: `TensorEngine` policy of `kv_input_row_offsets`
            (inferred).
        config: The MHA configuration with head count, depth, and swizzle
            mode used to build the `Depth512SM100Config`.
        group: Number of query heads per KV head for grouped-query attention.
        ragged: Whether the batch uses ragged sequence lengths with a
            non-null `valid_length` pointer.
        _is_cache_length_accurate: Whether the per-batch cache length values
            are exact.

    Args:
        output: Device buffer that receives the attention output.
        q_arg: Pointer to the query tensor data.
        k: Key operand providing the paged KV cache tile factory.
        v: Value operand providing the paged KV cache tile factory.
        num_rows_q: Number of query rows to process.
        mask: Causal or padding mask applied to the attention scores.
        valid_length: Per-batch pointer to valid cache lengths (used when
            `ragged` is true).
        max_prompt_len_arg: Maximum prompt length, static or runtime.
        max_cache_valid_length_arg: Maximum valid cache length across the
            batch.
        scale: Scaling factor applied to the QK dot product.
        kv_input_row_offsets: Optional ragged row-offset tensor for KV
            input rows.
        batch_size_arg: Number of sequences in the batch.
        partition: Partition scheme for the KV cache.
        ctx: Device context used to create TMA descriptors and enqueue the
            kernel.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match."
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    comptime assert not decoding, "depth512 pair-CTA does not support decoding"

    comptime d512_config = Depth512SM100Config[KVType.dtype](
        num_q_heads=config.num_heads,
        group=group,
        qk_depth=config.depth,
        ov_depth=config.depth,
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
    )
    comptime assert d512_config.supported(), d512_config.description()
    comptime swizzle_mode = d512_config.swizzle_mode
    # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
    # Q/K/V/S/P buffers governed by `swizzle_mode`).
    comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
    comptime fuse_gqa = d512_config.fuse_gqa
    comptime PairBM_eff = d512_config.BM_eff() * 2
    comptime num_threads = d512_config.num_threads  # 384

    var q = q_arg.bitcast[Scalar[KVType.dtype]]().unsafe_origin_cast[
        q_arg.origin
    ]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    # ---- TMA tile descriptors ------------------------------------------------

    # Output store: BM per CTA, full ov_depth. Single issuer, no combine
    # (depth_splits=1) -> one batched rank-5 TMA over the full depth (group==1).
    comptime store_blocks_per_op = o_store_tma_blocks_per_op[
        output_type,
        output_swizzle_mode,
        d512_config.ov_depth,
        d512_config.group if fuse_gqa else 1,
        depth_splits=1,
    ]()
    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=d512_config.BM,
        BN=d512_config.ov_depth,
        middle_dim=d512_config.num_kv_heads if fuse_gqa else d512_config.num_q_heads,
        group=d512_config.group if fuse_gqa else 1,
        tma_blocks_per_op=store_blocks_per_op,
    ]
    var ragged_tma_store = RaggedStoreType.create(
        ctx,
        output.unsafe_ptr(),
        rows=num_rows_q,
    )

    # Q: BM per CTA (not halved like 2Q).
    var q_tma_op = q_tma[
        swizzle_mode,
        BM=d512_config.BM,
        depth=d512_config.qk_depth,
        q_num_heads=d512_config.num_q_heads,
        group=d512_config.group,
        decoding=False,
        fuse_gqa=fuse_gqa,
        num_qk_stages=d512_config.num_qk_stages,
    ](ctx, q, num_rows_q)

    # K: each CTA loads BN//2 rows, BK0 depth per stage.
    comptime k_sub_BN = kv_sub_tile_rows(d512_config.BN // 2, KVType.page_size)
    # Depth-chunk TMA fold (SM100 K-only): fold the BK0 = num_chunks*gran depth
    # chunks into one rank-4 TMA when byte-equivalent. `kv_tma_fold_chunks` is the
    # single source of truth shared with the `tma_copy_k` issue site; here
    # box_rows == k_sub_BN and smem_BN == BN//2, so the fold is allowed exactly
    # when k_sub_BN == BN//2 (i.e. page_size >= BN//2 -> pages_per_iter == 1).
    comptime k_fold_chunks = kv_tma_fold_chunks[
        KVType.dtype,
        d512_config.swizzle_mode,
        BK=d512_config.BK0,
        head_size=d512_config.qk_depth,
        box_rows=k_sub_BN,
        smem_BN=d512_config.BN // 2,
        page_size=KVType.page_size,
    ]()
    var k_tma_op = k.create_tma_tile[
        d512_config.swizzle_mode,
        BN=k_sub_BN,
        depth=d512_config.qk_depth,
        BK=d512_config.BK0,
        fold_chunks=k_fold_chunks,
    ](ctx)

    # V: BK1 rows x v_cols_per_cta columns (heavily sub-tiled for SMEM).
    # Depth-chunk TMA fold (SM100): fold the v_cols_per_cta = num_chunks*gran
    # depth columns into one rank-4 TMA when byte-equivalent. Shares
    # `kv_tma_fold_chunks` with the `tma_copy_v` issue site (single source of
    # truth); here box_rows == v_sub_BN and smem_BN == BK1 (V's per-sub-tile
    # tile_rows), so the fold is allowed exactly when v_sub_BN == BK1 (i.e.
    # page_size >= BK1 -> pages_per_iter == 1).
    comptime v_sub_BN = kv_sub_tile_rows(d512_config.BK1, KVType.page_size)
    comptime v_fold_chunks = kv_tma_fold_chunks[
        KVType.dtype,
        d512_config.swizzle_mode,
        BK=d512_config.v_cols_per_cta,
        head_size=d512_config.ov_depth,
        box_rows=v_sub_BN,
        smem_BN=d512_config.BK1,
        page_size=KVType.page_size,
    ]()
    var v_tma_op = v.create_tma_tile[
        d512_config.swizzle_mode,
        BN=v_sub_BN,
        depth=d512_config.ov_depth,
        BK=d512_config.v_cols_per_cta,
        fold_chunks=v_fold_chunks,
    ](ctx)

    # ---- Scheduler -----------------------------------------------------------

    comptime SchedulerType = TransientScheduler[
        UInt32(PairBM_eff),
        UInt32(
            d512_config.num_kv_heads if fuse_gqa else d512_config.num_q_heads
        ),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
        pair_cta=True,
    ]
    var scheduler: SchedulerType = SchedulerType()

    # ---- Nested closure dispatch (no sink) -----------------------------------

    @__parameter
    @always_inline
    def with_kv_offsets[
        KVRowOffsetsType: OptionalPointer
    ](kv_row_offsets: KVRowOffsetsType) raises:
        @__parameter
        @always_inline
        def with_valid_length[
            ValidLengthType: OptionalPointer
        ](valid_len: ValidLengthType) raises:
            comptime PackType = Pack[
                MaskType,
                SchedulerType,
                ValidLengthType,
                NullPointer[.float32],  # no sink
                KVRowOffsetsType,
                MaxPromptLenType,
                PartitionType,
            ]
            var pack: PackType = {
                mask,
                scheduler,
                valid_len,
                NullPointer[.float32](),
                kv_row_offsets,
                max_prompt_len_arg,
                partition,
            }

            var max_num_prompt_tiles: UInt32 = ceildiv(
                max_prompt_len_arg.as_uint32(), UInt32(PairBM_eff)
            )
            var block_x: UInt32 = max_num_prompt_tiles
            # SchedulerType.grid_dim doubles block_x (pair_cta=True).

            logger.info("------ Dispatching to SM100 Depth512 Pair-CTA ------")
            logger.info(
                "QKV Type:",
                KVType.dtype,
                "Depth:",
                d512_config.qk_depth,
                "Number of Q // KV Heads:",
                d512_config.num_q_heads,
                "//",
                d512_config.num_kv_heads,
                "Batch Size:",
                batch_size,
                "Max Num Prompt Tiles:",
                max_num_prompt_tiles,
            )

            comptime smem_use = d512_config.smem_used

            comptime kernel = SM100MHADepth512[
                KVType,
                output_type,
                MaskType,
                SchedulerType,
                d512_config,
                ValidLengthType,
                KVRowOffsetsType,
                _is_cache_length_accurate,
                MaxPromptLenType,
                PartitionType,
            ].kernel

            ctx.enqueue_function[kernel](
                q_tma_op,
                k_tma_op,
                v_tma_op,
                ragged_tma_store,
                k,
                scale,
                batch_size,
                max_cache_valid_length,
                pack,
                grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                block_dim=(num_threads, 1, 1),
                cluster_dim=Dim(2, 1, 1),
                shared_mem_bytes=smem_use,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(smem_use)
                ),
            )

        # --- ragged dispatch ---
        comptime if ragged:
            with_valid_length[NonNullPointer[.uint32]](
                {valid_length.as_imm().as_unsafe_any_origin()}
            )
        else:
            with_valid_length[NullPointer[.uint32]]({})

    # --- kv_input_row_offsets dispatch ---
    if kv_input_row_offsets:
        with_kv_offsets[NonNullPointer[.uint32]](
            {kv_input_row_offsets.value().ptr}
        )
    else:
        with_kv_offsets[NullPointer[.uint32]]({})
