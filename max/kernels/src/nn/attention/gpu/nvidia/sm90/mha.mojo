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
"""SM90 (Hopper) FlashAttention-3 multi-head attention kernel and dispatch layer.

This module implements the warp-specialized, TMA-based MHA kernel for NVIDIA
H100 GPUs, along with the dispatch chain that materializes comptime
configuration choices (scheduler, sink, KV row offsets, ragged valid lengths)
into concrete kernel instantiations.
"""

from std.math import ceildiv, exp2, recip
from std.math.uutils import umod, uceildiv
from std.math.constants import log2e
from std.sys import align_of, get_defined_int, simd_width_of

import max.gpu.primitives.warp as warp
from std.collections import OptionalReg
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_dim,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import H100
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import external_memory
from max.gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor, TensorEngine
from layout.layout_tensor import copy_sram_to_dram
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
)
from nn.attention.mha_operand import kv_sub_tile_rows as _kv_sub_tile_rows
from nn.attention.gpu.nvidia.sm90.attention import (
    _apply_mask,
    _get_position,
    get_q_head_idx,
    ImmutTileTensor1D,
    KVTMATile,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem,
    Pack,
    produce,
    q_tma,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    QueuedTileScheduler,
    SeqInfo,
    TileScheduler,
    TransientScheduler,
)
from nn.attention.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)

from std.utils.index import Index
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple


@always_inline
def mha_sm90_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    KVRowOffsetsEngine: TensorEngine,
    SinkEngine: TensorEngine,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: DeviceBuffer[q_type],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask_functor: MaskType,
    valid_length: DeviceBuffer[.uint32],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        ImmutTileTensor1D[.uint32, Engine=KVRowOffsetsEngine]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type, Engine=SinkEngine]],
) raises:
    """Dispatches the SM90 FlashAttention-3 MHA kernel for a single batch.

    Selects a transient, tiled, or queued tile scheduler based on the
    persistent-kernel configuration, builds the Q/K/V TMA tile descriptors,
    and forwards the request down the dispatch chain to the enqueued kernel.

    Parameters:
        q_type: The dtype of the query tensor (inferred).
        KVType: The K/V operand type encoding dtype, page size, and layout
            (inferred).
        MaskType: The mask functor type applied to attention scores
            (inferred).
        output_type: The dtype of the output tensor (inferred).
        MaxPromptLenType: The maximum prompt length, possibly known at
            compile time (inferred).
        PartitionType: The scheme for partitioning attention work across
            SMs (inferred).
        KVRowOffsetsEngine: The `TensorEngine` of `kv_input_row_offsets`
            (inferred).
        SinkEngine: The `TensorEngine` of `sink_weights` (inferred).
        config: The MHA configuration holding block sizes, head count,
            depth, and algorithm.
        group: The query grouping factor, the number of query heads per
            KV head.
        ragged: Whether per-row valid lengths vary and require ragged
            masking.
        sink: Whether attention sink weights are applied.
        _is_cache_length_accurate: Whether the supplied cache length is
            exact and needs no clamping.

    Args:
        output: The device buffer that receives the attention output.
        q_arg: The device buffer holding the query tensor.
        k: The key operand lookup table.
        v: The value operand lookup table.
        num_rows_q: The number of query rows in the batch.
        mask_functor: The mask functor instance applied to attention
            scores.
        valid_length: The per-batch valid sequence length buffer.
        max_prompt_len_arg: The maximum prompt length after padding.
        max_cache_valid_length_arg: The maximum valid length of the KV
            cache.
        scale: The scaling factor applied to the QK^T product.
        kv_input_row_offsets: Optional row offsets into the KV cache for
            ragged layouts.
        batch_size_arg: The number of sequences in the batch.
        partition: The partition instance describing the work split.
        ctx: The device context used to enqueue the kernel.
        sink_weights: Optional sink weights tensor applied when sink is
            enabled.
    """
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    comptime swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    var q = (
        q_arg.unsafe_ptr()
        .bitcast[Scalar[KVType.dtype]]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )

    comptime decoding: Bool = MaxPromptLenType.static_value.or_else(0) == 1
    comptime new_config = MHAConfig[config.dtype](
        config.num_heads,
        config.depth,
        num_queries_per_block=64,
        num_keys_per_block=config.num_keys_per_block,
        BK=config.BK,
    ) if decoding else config
    comptime BM = new_config.block_m()
    comptime BK = new_config.padded_depth
    comptime assert BM % 64 == 0, "SM90 requires BM%64==0, but BM==" + String(
        BM
    )
    comptime assert (
        BK % 64 == 0
    ), "H100 requires BK%64==0 as it uses 128B swizzles, but BK==" + String(BK)
    comptime BN = new_config.block_n()
    # we add smem use for SharedMemBarrier synchronization
    # add the number of producer threads (i.e. 1 WARP_GROUP_SIZE)
    comptime num_threads = new_config.num_threads[True]()
    comptime assert num_threads % 128 == 0

    # Persistent kernels not currently supported with partitioning
    # This doesn't seem useful: we partition to make SMs more busy,
    # implying we don't have enough to make them persistent.
    # This also requires some tricky control flow handling to support,
    # which we haven't added yet.
    comptime persistent = 0 if PartitionType.do_partition else get_defined_int[
        "USE_EXPERIMENTAL_KERNELS", 0
    ]()
    comptime assert new_config.algorithm == FlashAttentionAlgorithm(3)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    # var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    # var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    # var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    comptime q_num_heads: Int = new_config.num_heads
    comptime num_scheduler_heads = q_num_heads // group if decoding else q_num_heads
    # if decoding,
    comptime scheduler_tile_shape = 1 if decoding else BM
    var q_tma_op = rebind[
        QTMATile[
            KVType.dtype,
            swizzle_mode,
            BM=new_config.block_m(),
            depth=new_config.depth,
            group=group,
            decoding=_is_decoding[MaxPromptLenType](),
        ]
    ](
        q_tma[
            swizzle_mode,
            BM=BM,
            depth=new_config.depth,
            q_num_heads=new_config.num_heads,
            group=group,
            decoding=decoding,
        ](ctx, q, num_rows_q)
    )
    comptime kv_sub_BN = _kv_sub_tile_rows(
        new_config.block_n(), KVType.page_size
    )
    var k_tma_op = k.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)
    var v_tma_op = v.create_tma_tile[
        swizzle_mode,
        BN=kv_sub_BN,
        depth=new_config.depth,
        BK=new_config.padded_depth,
    ](ctx)

    # materialize scheduler, call max prompt len
    comptime if persistent == 0:
        comptime SchedulerType = TransientScheduler[
            UInt32(scheduler_tile_shape),
            UInt32(num_scheduler_heads),
            flip_prompt_idx=not decoding
            and MaskType.get_type_name() == "CausalMask",
        ]
        var scheduler: SchedulerType = SchedulerType()
        _mha_sm90_sink_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            KVRowOffsetsEngine=KVRowOffsetsEngine,
            SinkEngine=SinkEngine,
            config=new_config,
            group=group,
            ragged=ragged,
            sink=sink,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            rebind[
                OptionalReg[ImmutTileTensor1D[KVType.dtype, Engine=SinkEngine]]
            ](sink_weights),
            partition,
            mask_functor,
            ctx,
        )
    elif persistent == 2:
        comptime SchedulerType = TileScheduler[
            UInt32(scheduler_tile_shape), UInt32(num_scheduler_heads)
        ]
        var scheduler: SchedulerType = SchedulerType()
        _mha_sm90_sink_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            KVRowOffsetsEngine=KVRowOffsetsEngine,
            SinkEngine=SinkEngine,
            config=new_config,
            group=group,
            ragged=ragged,
            sink=sink,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            rebind[
                OptionalReg[ImmutTileTensor1D[KVType.dtype, Engine=SinkEngine]]
            ](sink_weights),
            partition,
            mask_functor,
            ctx,
        )
    else:
        comptime SchedulerType = QueuedTileScheduler[
            UInt32(scheduler_tile_shape),
            UInt32(num_scheduler_heads),
            decoding=decoding,
        ]
        var schedule = ctx.enqueue_create_buffer[.uint32](1)
        schedule.enqueue_fill(UInt32(H100.sm_count))
        ctx.synchronize()
        var scheduler: SchedulerType = SchedulerType(
            schedule.unsafe_ptr().as_unsafe_any_origin()
        )
        _mha_sm90_sink_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            KVRowOffsetsEngine=KVRowOffsetsEngine,
            SinkEngine=SinkEngine,
            config=new_config,
            group=group,
            ragged=ragged,
            sink=sink,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            rebind[
                OptionalReg[ImmutTileTensor1D[KVType.dtype, Engine=SinkEngine]]
            ](sink_weights),
            partition,
            mask_functor,
            ctx,
        )
        _ = schedule


# materializes max prompt len, call partition
@always_inline
def _mha_sm90_sink_dispatch[
    SchedulerType: MHATileScheduler,
    KVLUTType: MHAOperand,
    output_type: DType,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    MaskType: MHAMask,
    KVRowOffsetsEngine: TensorEngine,
    SinkEngine: TensorEngine,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: DeviceBuffer[.uint32],
    kv_input_row_offsets: OptionalReg[
        ImmutTileTensor1D[.uint32, Engine=KVRowOffsetsEngine]
    ],
    sink_weights: OptionalReg[
        ImmutTileTensor1D[KVLUTType.dtype, Engine=SinkEngine]
    ],
    partition: PartitionType,
    mask: MaskType,
    ctx: DeviceContext,
) raises:
    comptime if sink:
        comptime SinkType = NonNullPointer[KVLUTType.dtype]
        var sink_ptr: SinkType = {sink_weights.value().ptr}
        _mha_sm90_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsEngine=KVRowOffsetsEngine,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            ctx,
        )
    else:
        comptime SinkType = NullPointer[KVLUTType.dtype]
        comptime sink_ptr: SinkType = {}
        _mha_sm90_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsEngine=KVRowOffsetsEngine,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            ctx,
        )


# materializes sink, calls kv_input_row_offsets

# materializes partition, calls sink # not real
# materializes kv_input_row_offsets, calls kernel


@always_inline
def _mha_sm90_kv_input_row_offset_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    SinkType: OptionalPointer,
    KVRowOffsetsEngine: TensorEngine,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: DeviceBuffer[.uint32],
    kv_input_row_offsets: OptionalReg[
        ImmutTileTensor1D[.uint32, Engine=KVRowOffsetsEngine]
    ],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    ctx: DeviceContext,
) raises:
    comptime KVRowOffsetsNonNull = NonNullPointer[.uint32]
    comptime KVRowOffsetsNull = NullPointer[.uint32]
    if kv_input_row_offsets:
        var kv_row_offsets: KVRowOffsetsNonNull = {
            kv_input_row_offsets.value().ptr
        }
        _mha_sm90_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNonNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            ctx,
        )
    else:
        var kv_row_offsets: KVRowOffsetsNull = {}
        _mha_sm90_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            ctx,
        )


@always_inline
def _mha_sm90_valid_length_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: DeviceBuffer[.uint32],
    kv_input_row_offsets: KVRowOffsetsType,
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    ctx: DeviceContext,
) raises:
    comptime if ragged:
        comptime ValidLengthType = NonNullPointer[.uint32]
        var valid_len: ValidLengthType = {valid_length}
        _mha_sm90_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            ctx,
        )
    else:
        comptime ValidLengthType = NullPointer[.uint32]
        var valid_len: ValidLengthType = {}
        _mha_sm90_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            config=config,
            group=group,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            ctx,
        )


@always_inline
def _mha_sm90_enqueue[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_ptr_arg: DeviceBuffer[output_type],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: ValidLengthType,  # OptionalPointer[DType.uint32]
    kv_input_row_offsets: KVRowOffsetsType,  # OptionalPointer[DType.uint32],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    ctx: DeviceContext,
) raises:
    # the pack contains all possibly 0-sized objects
    comptime kernel_sm90 = _mha_sm90[
        KVLUTType,
        output_type,
        MaskType,
        SchedulerType,
        config,
        group,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        _is_cache_length_accurate,
        MaxSeqLenType,
        PartitionType,
        swizzle_mode,
    ]
    comptime PackType = Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ]
    var pack: PackType = {
        mask,
        scheduler,
        valid_length,
        sink_weights,
        kv_input_row_offsets,
        max_seq_len,
        partition,
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_seq_len.as_uint32(), UInt32(config.block_m())
    )
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    comptime smem_use = config.shared_mem_bytes[True, sm_90=True]()
    comptime num_threads = config.num_threads[True]()
    ctx.enqueue_function[kernel_sm90](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_ptr_arg,
        kv_lut,
        scale,
        batch_size,
        num_keys_arg,
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, block_x),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )


@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads[True]())
    )
)
@__name(
    t"sm90_mha_depth{config.depth}_{KVLUTType.dtype}_{output_type}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def _mha_sm90[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM=config.block_m(),
        depth=config.depth,
        group=group,
        decoding=_is_decoding[MaxSeqLenType](),
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BN=_kv_sub_tile_rows(config.block_n(), KVLUTType.page_size),
        BK=config.padded_depth,
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    num_keys_arg: UInt32,
    pack: Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ],
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO). The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    comptime kv_type = KVLUTType.dtype
    comptime decoding: Bool = _is_decoding[MaxSeqLenType]()

    comptime simd_size = simd_width_of[kv_type]()
    comptime ragged = not ValidLengthType.is_null

    comptime num_warps_m = config.num_warps_m()
    comptime num_consumer_threads = config.num_consumer_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    # num_consumer_threads ignores the producers
    # actual number of threads is num_consumer_threads + 128
    comptime num_consumer = num_consumer_threads // WARPGROUP_SIZE
    comptime pipeline_stages = config.num_pipeline_stages
    var tid = UInt32(thread_idx.x)
    var warp_group_idx: UInt32 = warp.broadcast(tid // UInt32(WARPGROUP_SIZE))

    var mask = pack.mask
    var scheduler = pack.scheduler
    var valid_length = pack.valid_length
    var sink_weights = pack.sink_weights
    var kv_input_row_offsets = pack.kv_input_row_offsets
    var max_seq_len = pack.max_seq_len
    var partition = pack.partition

    comptime assert num_warps_m == (
        num_consumer_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    var warp_id: UInt32 = warp.broadcast(
        (tid - UInt32(WARPGROUP_SIZE)) // UInt32(WARP_SIZE)
    )
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y: UInt32 = warp_id  # // num_warps_n

    comptime q_smem_layout_consumer = tile_layout_k_major[
        DType.bfloat16,
        BM,
        config.padded_depth,
        swizzle_mode=swizzle_mode,
    ]()
    comptime k_smem_layout = tile_layout_k_major[
        DType.bfloat16,
        BN,
        config.padded_depth,
        swizzle_mode=swizzle_mode,
    ]()
    comptime v_smem_layout = tile_layout_mn_major[
        DType.bfloat16,
        config.padded_depth,
        BN,
        swizzle_mode=swizzle_mode,
    ]()
    # for wgmma_0, we multiply BM x depth @ depth x BN -> BM x BN
    # for wgmma_1, we multiply BM x BN @ BN x depth -> BM x depth
    # For wgmma_0, we iterate over (depth//BK) tiles of size BKxBN
    # For wgmma_1, we iterate over (BN//BK) tiles of size BKxdepth
    comptime persistent = SchedulerType.may_advance

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime q_size = q_smem_layout_consumer.size()
    comptime q_smem_size = 2 * q_size if persistent else q_size
    var q_smem = external_memory[
        Scalar[kv_type],
        address_space=.SHARED,
        alignment=128,
        name="mha_dynamic_shared_memory",
    ]()
    # We have `num_pipeline_stages` instances of each
    comptime kv_smem_size = config.kv_smem_size(True)
    var kv_smem = q_smem + q_smem_size

    # var head_idx: UInt32 = block_idx.y
    # var q_tile_idx: UInt32 = block_idx.x

    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime MMA_M = 16  # per warp
    comptime MMA_N0 = BN
    comptime MMA_N1 = config.padded_depth
    comptime MMA_K = 16
    comptime WM = config.WM
    comptime num_m_mmas = WM // MMA_M
    comptime assert num_m_mmas == 1, "FIXME: life this constraint"
    # alias WN = config.WN
    # alias num_n_mmas = WN // MMA_N
    comptime num_n_mmas = 1
    # alias num_k_mmas = BK // MMA_K

    comptime accum_type = get_accum_type[kv_type]()
    comptime assert (
        accum_type.is_floating_point()
    ), "accum_type must be floating point"
    comptime p_frag_size = MMA_M * MMA_N0 // WARP_SIZE
    comptime o_frag_size = MMA_M * MMA_N1 // WARP_SIZE
    comptime frag_simdwidth = 2

    comptime a_frag_size = MMA_M * MMA_K // WARP_SIZE
    # MMA_N0 // MMA_K
    comptime frag_ratio = p_frag_size // a_frag_size

    # the first mma is BMxdepth @ depthxBN
    comptime wgmma_0 = TensorCoreAsync[
        accum_type,
        kv_type,
        kv_type,
        Index(4 * MMA_M, MMA_N0, 16),
        a_swizzle=swizzle_mode,
        b_swizzle=swizzle_mode,
        transpose_b=True,
    ]()
    # the second mma is BMxBN @ BNxdepth
    comptime wgmma_1 = TensorCoreAsync[
        accum_type,
        kv_type,
        kv_type,
        Index(4 * MMA_M, MMA_N1, 16),
        a_swizzle=TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle_mode,
        transpose_b=False,
    ]()

    comptime num_row_blocks_per_mma = 2
    # a wgmma.m64n32k16 `D` fragment looks like
    #
    # 0,1  4,5   8, 9  12,13
    # 2,3  6,7  10,11  14,15
    #
    # Each row/column has `p_frag_simdwidth`-sized vectors
    # (e.g. `4,5` is of size 2 = p_frag_simdwidth)
    # We have `num_row_blocks_per_mma` rows.
    # The total number of elements (16) equals `p_frag_size`.
    # The number of columns equals
    # `p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth)`
    #
    # This gives us the layout:
    #
    # Note the ordering of strides:
    # ((1, 3), (0, 2, 4))
    # alias output_layout = Layout(
    #     IntTuple(
    #         IntTuple(num_row_blocks_per_mma, num_m_mmas),
    #         IntTuple(
    #             p_frag_simdwidth,
    #             p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
    #             num_n_mmas,
    #         ),
    #     ),
    #     IntTuple(
    #         IntTuple(p_frag_simdwidth, p_frag_size),
    #         IntTuple(1, 2 * p_frag_simdwidth, num_m_mmas * p_frag_size),
    #     ),
    # )
    # Vectorizing the layout:
    comptime element_layout = Layout.row_major(1, frag_simdwidth)
    comptime vec_output_row_shape = IntTuple(num_row_blocks_per_mma, num_m_mmas)
    comptime p_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    comptime o_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                o_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, o_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * o_frag_size,
            ),
        ),
    )
    comptime num_rows_per_warp = p_vec_output_layout[0].size()
    comptime num_cols_p = p_vec_output_layout[1].size()
    comptime num_cols_output = o_vec_output_layout[1].size()

    # Rowwise max and sum for online softmax
    comptime accum_simd_width = simd_width_of[accum_type]()
    comptime row_alignment = align_of[SIMD[accum_type, accum_simd_width]]()

    comptime mma_thread_layout = Layout.row_major(8, 4)

    # Handle sink_weights
    var sink_weights_ptr = Optional[
        UnsafePointer[Scalar[kv_type], ImmutAnyOrigin]
    ]()

    comptime if not SinkType.is_null:
        sink_weights_ptr = rebind[
            UnsafePointer[Scalar[kv_type], ImmutAnyOrigin]
        ](sink_weights.value())

    var produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    var consumed_mbar_kv = produced_mbar_kv + pipeline_stages
    var produced_mbar_q = consumed_mbar_kv + pipeline_stages
    var consumed_mbar_q = produced_mbar_q + 2
    var block_idx_ptr = (consumed_mbar_q + 2).bitcast[UInt32]()

    # comptime USE_TMA = True
    comptime USE_TMA = False
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    comptime num_producer_regs = 56 if num_consumer == 1 else (
        (24 if USE_TMA else 56) if num_consumer == 2 else 32
    )
    comptime num_consumer_regs = 256 if num_consumer == 1 else (
        (240 if USE_TMA else 224) if num_consumer == 2 else 160
    )
    # alias num_producer_regs = 56
    # alias num_consumer_regs = 224

    # constructing calls barrier() if static
    var tile_summary = MHATileSummary[ValidLengthType](
        batch_size,
        ceildiv(max_seq_len.as_uint32(), UInt32(BM))
        * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    var state: MHATileState = scheduler.initial_state(
        block_idx_ptr.as_unsafe_any_origin(), tile_summary
    )

    # returns `true` if we are done
    @__parameter
    @always_inline
    def advance[
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](pipeline_idx: UInt32) -> OptionalReg[SeqInfo]:
        return scheduler.advance[producer=producer, sync=sync](
            tile_summary, state, pipeline_idx
        )

    # The persistent kernels limit the grid size.
    # initial_seq_info = scheduler.unsafe_get_current_work_info(tile_summary, state)

    var initial_seq_info = scheduler.unsafe_seq_info(tile_summary, state)

    comptime if not decoding:
        if not initial_seq_info.is_valid():
            comptime if persistent:
                var seq_info = advance[True, MHASchedulerSynchronization.ALL](1)
                if seq_info:
                    initial_seq_info = seq_info.value()
                else:
                    return
            else:
                return

    if tid == 0:
        comptime for i in range(pipeline_stages):
            produced_mbar_kv[i].init(1)
            consumed_mbar_kv[i].init(Int32(num_consumer_threads))

        comptime if persistent:
            comptime for i in range(2):
                produced_mbar_q[i].init(1)
                consumed_mbar_q[i].init(Int32(num_consumer_threads))

    comptime PositionType = MHAPosition[
        BM,
        BN,
        config.depth,
        config.padded_depth,
        num_heads,
        group,
        decoding,
    ]

    @__parameter
    @always_inline
    def k_tile(
        idx: UInt32,
        out k_smem: LayoutTensor[
            kv_type,
            k_smem_layout,
            MutAnyOrigin,
            address_space=.SHARED,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            alignment=128,
        ],
    ):
        comptime sz = BN * config.padded_depth
        k_smem = {(kv_smem + UInt32(sz) * idx).as_unsafe_any_origin()}

    @__parameter
    @always_inline
    def v_tile(
        idx: UInt32,
        out v_smem: LayoutTensor[
            kv_type,
            v_smem_layout,
            MutAnyOrigin,
            address_space=.SHARED,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            alignment=128,
        ],
    ):
        comptime sz = BN * config.padded_depth
        v_smem = {(kv_smem + UInt32(sz) * idx).as_unsafe_any_origin()}

    @__parameter
    @always_inline
    def get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM,
            BN,
            config.depth,
            config.padded_depth,
            num_heads,
            group,
            ragged,
            _is_cache_length_accurate,
        ](
            seq_info,
            kv_lut,
            max_seq_len,
            num_keys_arg,
            kv_input_row_offsets,
        )

    var position: PositionType = get_position(initial_seq_info)

    var q_pipeline_state = PipelineState[2]()

    barrier()
    # For intra-warp overlap, we initiate wgmmas as
    # Q @ K_0, Q @ K_1, P_0 @ V_0, Q @ K_2, P_1 @ V_1, ...
    # ..., Q @ K_{N-1}, P_{N-2} @ V_{N-2}, P_{N-1} @ V_{N-1}
    #
    # Due to this, we can overlap wgmmas and softmax calculations.
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        if thread_idx.x == 0:
            produce[
                swizzle_mode,
                pipeline_stages=pipeline_stages,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                v_tma_op,
                q_smem.as_unsafe_any_origin(),
                kv_smem.as_unsafe_any_origin(),
                produced_mbar_kv.as_unsafe_any_origin(),
                consumed_mbar_kv.as_unsafe_any_origin(),
                produced_mbar_q.as_unsafe_any_origin(),
                consumed_mbar_q.as_unsafe_any_origin(),
                kv_lut,
                position,
                partition,
                scheduler,
                mask,
                tile_summary,
                state,
                max_seq_len,
                num_keys_arg,
                kv_input_row_offsets,
            )

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # arrive to unblock the producers
        # TODO: skip this by not waiting on the first set
        comptime for i in range(pipeline_stages):
            _ = consumed_mbar_kv[i].arrive()

        comptime if persistent:
            _ = consumed_mbar_q[0].arrive()
        var local_warp_group_idx: UInt32 = warp_group_idx - 1

        @__parameter
        @always_inline("nodebug")
        def q_consumer(
            q_idx: UInt32,
        ) -> LayoutTensor[
            kv_type,
            q_smem_layout_consumer,
            MutAnyOrigin,
            address_space=.SHARED,
            alignment=128,
        ]:
            return {(q_smem + UInt32(q_size) * q_idx).as_unsafe_any_origin()}

        # layout is
        # shape  = (2, num_m_mmas) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)
        comptime s_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas, p_frag_size
        )
        comptime o_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas, o_frag_size
        )
        var p_reg_tile = LayoutTensor[
            accum_type,
            s_reg_tile_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation()
        var output_reg_tile = (
            LayoutTensor[
                accum_type,
                o_reg_tile_layout,
                MutAnyOrigin,
                address_space=.LOCAL,
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime p_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
        )
        var p_frag = LayoutTensor[
            kv_type,
            p_reg_tile_layout,
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation()

        @__parameter
        @always_inline
        def vectorize_p_reg_tile(
            out result: LayoutTensor[
                accum_type,
                p_vec_output_layout,
                MutAnyOrigin,
                address_space=.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = {p_reg_tile.ptr}

        @__parameter
        @always_inline
        def vectorize_o_reg_tile(
            out result: LayoutTensor[
                accum_type,
                o_vec_output_layout,
                MutAnyOrigin,
                address_space=.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = {output_reg_tile.ptr}

        var rowmax = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation()
        var rowsum = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutAnyOrigin,
            address_space=.LOCAL,
        ].stack_allocation()

        # Mask global memory iterator.
        var mask_warp_row = warp_y * UInt32(WM)
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[
                accum_type
            ]() if MaskType.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )

        @__parameter
        @always_inline
        def q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
            var k_smem_sub = k_tile(read_idx)
            var q_smem_sub = q_consumer(q_idx)
            produced_mbar_kv[read_idx].wait(read_phase)

            warpgroup_fence(p_reg_tile)
            wgmma_0.arrive()
            wgmma_0.wgmma[
                num_consumer,
                scale_c=0,
                num_k_iters=Optional[Int](
                    uceildiv(depth, wgmma_0.mma_shape[2])
                ),
            ](
                q_smem_sub,
                k_smem_sub,
                p_reg_tile,
                Int(local_warp_group_idx),
            )
            wgmma_0.commit_group()
            warpgroup_fence(p_reg_tile)

        @__parameter
        @always_inline
        def p_mul_v(read_idx: UInt32, read_phase: UInt32):
            var v_smem_sub = v_tile(read_idx)
            produced_mbar_kv[read_idx].wait(read_phase)
            warpgroup_fence(output_reg_tile)
            wgmma_1.arrive()
            wgmma_1.wgmma(
                p_frag,
                v_smem_sub,
                output_reg_tile,
            )
            wgmma_1.commit_group()
            warpgroup_fence(output_reg_tile)

        @__parameter
        @always_inline
        def wait_for_q_mul_k[wgmma_left_in_flight: Int](read_idx: UInt32):
            wgmma_0.wait_group[wgmma_left_in_flight]()  # P is available
            _ = consumed_mbar_kv[read_idx].arrive()

        @__parameter
        @always_inline
        def wait_for_p_mul_v(read_idx: UInt32):
            wgmma_1.wait_group[0]()  # output is available
            _ = consumed_mbar_kv[read_idx].arrive()

        @__parameter
        @always_inline
        def apply_mask(
            position: PositionType,
            mask_status: TileMaskStatus,
            kv_tile_start_row: UInt32,
        ):
            var max_len: UInt32 = (
                num_keys_arg if decoding else max_seq_len.as_uint32()
            )
            _apply_mask[WM, MMA_N0, num_m_mmas, num_n_mmas](
                mask_warp_row,
                position,
                lane,
                max_len,
                scale_log2e,
                kv_tile_start_row,
                mask,
                mask_status,
                vectorize_p_reg_tile(),
            )

        @__parameter
        @always_inline
        def scale_output(correction: type_of(rowmax)):
            # we are now able to read/modify `output_reg_tile` and modify `p_frag`
            var vout = vectorize_o_reg_tile()

            # Correct output
            # We could avoid this on the first iter
            # if we specialize and unswitch on `first_iter`
            # otherwise, the branch requires synchronization
            comptime for row in range(num_rows_per_warp):
                var c = SIMD[accum_type, element_layout.size()](
                    rebind[Scalar[accum_type]](correction[row])
                )

                comptime for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * c

        @__parameter
        @always_inline
        def elementwise_reciprocal(
            old_rowsum: type_of(rowsum), new_rowsum: type_of(rowsum)
        ):
            # new_rowsum, old_rowsum = 1/old_rowsum, new_rowsum
            comptime for row in range(num_rows_per_warp):
                var old = old_rowsum[row]
                var new = new_rowsum[row]
                new_rowsum[row] = recip(old)[0]
                old_rowsum[row] = new

        @__parameter
        @always_inline
        def write_output(
            position: PositionType,
            q_idx: UInt32,
            rowsum_inv: type_of(rowsum),
        ):
            var vout = vectorize_o_reg_tile()

            # Apply softmax denumerator.
            comptime for row in range(num_rows_per_warp):
                var rs_inv = vout.element_type(rowsum_inv[row][0])

                comptime for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * rs_inv

            var output_ptr: UnsafePointer[
                Scalar[output_type], MutAnyOrigin
            ] = o_ptr_arg

            comptime if decoding and PartitionType.do_partition:
                output_ptr = output_ptr + (
                    UInt32(depth * num_heads)
                    * batch_size
                    * position.prompt_offset
                )
            var output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            comptime swizzle = make_swizzle[
                num_rows=MMA_M // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            comptime q_tile_size: UInt32 = UInt32(q_smem_size // 2)

            # ensure all threads have finished reading `q_smem`
            named_barrier[Int32(num_consumer_threads)]()
            var accum_smem_tile = output_reg_to_smem[
                BM,
                config.depth,
                config.padded_depth,
                swizzle,
                num_consumer,
            ](
                tid,
                local_warp_group_idx,
                warp_y,
                (q_smem + q_idx * q_tile_size).as_unsafe_any_origin(),
                output_reg_tile,
            )
            # Guard writing to shared memory.
            named_barrier[Int32(num_consumer_threads)]()
            # Vectorized copy from shared to global memory, during which every 2 FP32
            # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
            # vector and stored using 16B store instruction.
            copy_sram_to_dram[
                thread_layout=Layout.row_major(
                    num_consumer_threads * simd_size // config.depth,
                    config.depth // simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, simd_size](),
                accum_smem_tile.vectorize[1, simd_size](),
            )

        var startend = position.get_start_and_end_for_partitions[
            page_size=KVLUTType.page_size
        ](partition, mask)
        var kv_tile_start_row: UInt32 = startend[0]
        var end: UInt32 = startend[1]

        comptime if decoding and PartitionType.do_partition:
            if kv_tile_start_row >= end:
                if umod(thread_idx.x, 4) == 0 and thread_idx.x < (
                    4 * min(group, 8) + 128
                ):
                    var exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_heads = get_q_head_idx(position, lane)

                    comptime for i in range(q_heads.size):
                        var q_head_idx = q_heads[i]
                        exp_sum_ptr[q_head_idx] = Scalar[
                            PartitionType.accum_dtype
                        ](0)
                        qk_max_ptr[q_head_idx] = min_or_neg_inf[
                            PartitionType.accum_dtype
                        ]()

                write_output(position, q_pipeline_state.index(), rowsum)
                return

        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += UInt32(BN)

        var read_pipeline_states = PipelineState[pipeline_stages]()

        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        var read_idx_q: UInt32 = read_pipeline_states.index()
        q_mul_k(
            read_idx_q,
            read_pipeline_states.phase(),
            q_pipeline_state.index(),
        )
        read_pipeline_states.step()
        wait_for_q_mul_k[0](read_idx_q)

        apply_mask(position, mask_status, kv_tile_start_row)

        var sink_weight: Scalar[accum_type] = 0.0

        # Include sink_weights in rowmax computation if present
        comptime if not SinkType.is_null:
            var q_head_indices = get_q_head_idx(position, lane)

            comptime if decoding:
                comptime for i in range(q_head_indices.size):
                    var head_idx = q_head_indices[i]
                    sink_weight = (
                        sink_weights_ptr.unsafe_value()[head_idx].cast[
                            accum_type
                        ]()
                        * log2e
                    )
                    rowmax[i] = sink_weight
            else:
                sink_weight = (
                    sink_weights_ptr.unsafe_value()[q_head_indices[0]].cast[
                        accum_type
                    ]()
                    * log2e
                )

                comptime for i in range(num_rows_per_warp):
                    rowmax[i] = sink_weight

        # Compute initial rowmax
        var attention_rowmax = _rowmax_online_softmax[
            # threads layout by warp
            1,
            mma_thread_layout,
            use_exp2=True,
        ](vectorize_p_reg_tile(), rowmax, init_rowmax=SinkType.is_null)

        rowmax.copy_from(attention_rowmax)

        # Compute rowsum
        var attention_rowsum = _rowsum[mma_thread_layout](
            vectorize_p_reg_tile()
        )

        # Add sink weight contribution to rowsum
        comptime if not SinkType.is_null:
            var q_head_indices = get_q_head_idx(position, lane)

            comptime if decoding:
                comptime for i in range(q_head_indices.size):
                    var sink_contribution = exp2(sink_weight - rowmax[i])
                    attention_rowsum[i] += sink_contribution[0]

            else:
                comptime for i in range(num_rows_per_warp):
                    # Compute exp2((sink_weight - rowmax[j]) * log2e)
                    var sink_contribution = exp2(sink_weight - rowmax[i])
                    attention_rowsum[i] += sink_contribution[0]

        rowsum.copy_from(attention_rowsum)

        var position_prev: PositionType = position
        var q_idx_old: UInt32 = q_pipeline_state.index()
        var q_phase_old: UInt32 = q_pipeline_state.phase()

        # Consumption order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        comptime if persistent:
            kv_tile_start_row += UInt32(BN)
        while True:
            while True:
                comptime if not persistent:
                    kv_tile_start_row += UInt32(BN)
                if kv_tile_start_row >= end:
                    break

                # this loops over num_keys
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    comptime if persistent:
                        kv_tile_start_row += UInt32(BN)
                    continue
                p_frag.vectorize[
                    1, a_frag_size
                ]().copy_from(  # copy new pfrag, used by `p_mul_v` on next iter
                    p_reg_tile.reshape[
                        Layout.row_major(
                            num_m_mmas * num_n_mmas * frag_ratio,
                            a_frag_size,
                        )
                    ]().vectorize[1, a_frag_size](),
                )

                # new pipeline states
                var read_idx_q: UInt32 = read_pipeline_states.index()
                # start wgmmas
                q_mul_k(
                    read_idx_q,
                    read_pipeline_states.phase(),
                    q_pipeline_state.index(),
                )  # can't rw `p_reg_tile`
                read_pipeline_states.step()
                var read_idx_v: UInt32 = read_pipeline_states.index()
                p_mul_v(
                    read_idx_v, read_pipeline_states.phase()
                )  # can't rw output or pfrag
                read_pipeline_states.step()
                wait_for_q_mul_k[1](read_idx_q)  # can rw `p_reg_tile`

                apply_mask(position, mask_status, kv_tile_start_row)
                var new_q = persistent and q_idx_old != q_pipeline_state.index()
                # Compute rowmax for current scores
                var current_rowmax = _rowmax_online_softmax[
                    # threads layout by warp
                    1,
                    mma_thread_layout,
                    use_exp2=True,
                ](vectorize_p_reg_tile(), rowmax, new_q)

                var score_frag_rowmax = current_rowmax
                if new_q:
                    comptime if decoding and PartitionType.do_partition:
                        if umod(thread_idx.x, 4) == 0 and thread_idx.x < (
                            4 * min(group, 8) + 128
                        ):
                            var exp_sum_ptr, qk_max_ptr = (
                                position_prev.exp_sum_qk_max_ptr(
                                    partition, batch_size
                                )
                            )
                            var q_heads = get_q_head_idx(position, lane)

                            comptime for i in range(q_heads.size):
                                var q_head_idx = q_heads[i]
                                exp_sum_ptr[q_head_idx] = rebind[
                                    Scalar[PartitionType.accum_dtype]
                                ](rowsum[i])
                                qk_max_ptr[q_head_idx] = rebind[
                                    Scalar[PartitionType.accum_dtype]
                                ](rowmax[i])
                    var score_frag_rowsum = rebind[type_of(rowsum)](
                        _rowsum[mma_thread_layout](vectorize_p_reg_tile())
                    )
                    rowmax.copy_from(score_frag_rowmax)
                    elementwise_reciprocal(rowsum, score_frag_rowsum)
                    wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
                    # we `^ 1` to access the previous
                    # Two separate issues:
                    # 0. Which q do we use for `accum_smem`?
                    # 1. Which qs, if any, do we `arrive` at?
                    #
                    # If the next q_idx != the current q_idx (i.e. q_idx_n != q_idx)
                    # then we can use the current q for writing smem.
                    # If `q_idx_n == q_idx`, then we use the old q_idx (i.e. q_idx_o).
                    # This means we were not allowed to `arrive` at `q_idx_o`.
                    #
                    # Letting `0` indicate inequality, and `1` equality,
                    # let x = q_idx == q_idx_n
                    # let y = q_idx_n == q_idx_n_n
                    # We thus have 4 states `xy`:
                    # 0. 00: We use q_idx and arrive
                    # 1. 01: We use q_idx, but do not arrive on q_idx
                    # 2. 10: We use q_idx_o, do not arrive on q_idx
                    # 3. 11: We use q_idx_o, do not arrive on q_idx
                    #
                    # Only in `00` do we get to arrive on `q_idx` early.
                    # Given `BN < num_keys`, it won't often be the case
                    # that we can arrive at Q early; we need a series
                    # of q_tile_idx and head_idx that have a lot of
                    # `FULL_MASK`s, which our iteration scheme is supposed
                    # to make unlikely.
                    # Thus, we're going to simplify the problem by assuming
                    # scenario `0.` is unlikely unless `BN >= num_keys`,
                    # in which case it is guaranteed.
                    # var q_idx: UInt32 = q_pipeline_state.index() if few_keys else q_idx_old
                    write_output(position_prev, q_idx_old, score_frag_rowsum)
                    var q_idx_new: UInt32 = q_pipeline_state.index()

                    _ = consumed_mbar_q[q_idx_new].arrive()
                    _ = output_reg_tile.vectorize[accum_simd_width]().fill(0)
                    position_prev = position
                    q_idx_old = q_idx_new
                    q_phase_old = q_pipeline_state.phase()
                else:
                    var score_frag_rowsum = rebind[type_of(rowsum)](
                        _rowsum[mma_thread_layout](vectorize_p_reg_tile())
                    )

                    _online_softmax_correction[use_exp2=True](
                        rowmax, score_frag_rowmax
                    )
                    # rowmax now holds score_frag_rowmax
                    # score_frag_rowmax now holds the correction

                    comptime for i in range(num_rows_per_warp):
                        rowsum[i] = (
                            rowsum[i] * score_frag_rowmax[i]
                            + score_frag_rowsum[i]
                        )

                    wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
                    scale_output(score_frag_rowmax)  # scale output

                comptime if persistent:
                    kv_tile_start_row += UInt32(BN)

            comptime if persistent:
                var q_idx_old: UInt32 = q_pipeline_state.index()
                var q_phase_old: UInt32 = q_pipeline_state.phase()
                q_pipeline_state.step()
                produced_mbar_q[q_idx_old].wait(q_phase_old)
                var docontinue = advance[False](q_idx_old)
                if not docontinue:
                    break
                position = get_position(docontinue.value())
                var start, new_end = position.get_start_and_end_for_partitions[
                    page_size=KVLUTType.page_size
                ](partition, mask)
                kv_tile_start_row = start
                end = new_end
            else:
                break

        p_frag.vectorize[1, a_frag_size]().copy_from(
            p_reg_tile.reshape[
                Layout.row_major(
                    num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                )
            ]().vectorize[1, a_frag_size](),
        )
        p_mul_v(
            read_pipeline_states.index(),
            read_pipeline_states.phase(),
        )

        comptime if decoding and PartitionType.do_partition:
            if umod(thread_idx.x, 4) == 0 and thread_idx.x < (
                4 * min(group, 8) + 128
            ):
                var exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                    partition, batch_size
                )
                var q_heads = get_q_head_idx(position, lane)

                comptime for i in range(q_heads.size):
                    var q_head_idx = q_heads[i]
                    exp_sum_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowsum[i])
                    qk_max_ptr[q_head_idx] = rebind[
                        Scalar[PartitionType.accum_dtype]
                    ](rowmax[i])

        comptime for row in range(num_rows_per_warp):
            rowsum[row] = recip(rowsum[row])[0]
        wgmma_1.wait_group()
        write_output(position, q_pipeline_state.index(), rowsum)
        # don't arrive
