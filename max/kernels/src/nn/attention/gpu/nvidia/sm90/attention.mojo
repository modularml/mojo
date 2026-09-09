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

"""
SM90 (Hopper) multi-head attention producer-side helpers.

Re-exports shared NVIDIA attention primitives from `common` and provides the
SM90-specific implementations of `produce` (TMA-based Q/K/V tile staging),
`_apply_mask`, `_get_position`, `get_q_head_idx`, `output_reg_to_smem`, and
`_optional_lt_to_tt` consumed by the `sm90/mha.mojo` kernel.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.math.uutils import ufloordiv
from std.math.constants import log2e

from std.sys import size_of

import max.gpu.primitives.warp as warp
from std.algorithm.functional import unswitch
from max.gpu import thread_idx
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from layout.layout_tensor import copy_local_to_shared
from layout.swizzle import Swizzle
from layout.tensor_core_async import tile_layout_k_major
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import (
    MHAOperand,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
)
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
)
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple
from std.utils import StaticTuple

# Re-export shared NVIDIA attention primitives that now live in `common`, so
# the helpers kept in this file (`produce`, `_apply_mask`, `_get_position`,
# `get_q_head_idx`, `output_reg_to_smem`, `_optional_lt_to_tt`) and external
# sm90 consumers (e.g. `sm90/mha.mojo`) keep resolving them via this module.
from nn.attention.gpu.nvidia.common import (
    ImmutTileTensor1D,
    KVTMATile,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    QTMATile,
    _LocalTT,
    _SharedMemTT,
    elect,
    kv_coord,
    output_reg_to_smem_st_matrix,
    q_coord,
    q_tma,
)


@always_inline
def _optional_lt_to_tt[
    dtype: DType,
](
    opt: OptionalReg[
        LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ],
) -> OptionalReg[ImmutTileTensor1D[dtype]]:
    """Convert an OptionalReg[LayoutTensor] to OptionalReg[TileTensor]."""
    if opt:
        # NOTE: a plain `return lt_to_tt(opt.value())` compiles for the host
        # target but FAILS the sm90 GPU-target compile: there `lt_to_tt`'s
        # inferred result layout and `ImmutTileTensor1D`'s declared layout are
        # structurally identical but not type-identical (the 1-D contiguous
        # stride is `ComptimeInt[1]` on one side and a runtime `Int64` on the
        # other). The stride of a contiguous 1-D tensor is always 1, so rebind
        # to bridge the type-identity gap. Verify changes here with a remote
        # GPU build (`--config=remote-b200`), not just a host build.
        return rebind[ImmutTileTensor1D[dtype]](lt_to_tt(opt.value()))
    return None


@always_inline
def _get_position[
    KVLUTType: MHAOperand,
    MaxSeqLenType: OptionallyStaticInt,
    KVInputRowOffsetsType: OptionalPointer,
    //,
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    q_num_heads: Int,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    out ret: MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        q_num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ],
    seq_info: SeqInfo,
    kv_lut: KVLUTType,
    max_seq_len: MaxSeqLenType,
    num_keys_arg: UInt32,
    kv_input_row_offsets: KVInputRowOffsetsType,
):
    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_row: UInt32

    comptime if ragged:
        comptime if not _is_cache_length_accurate:
            start_pos = UInt32(
                warp.broadcast(kv_lut.cache_length(Int(batch_idx)))
            )
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        comptime if KVInputRowOffsetsType.is_null:
            num_keys = seq_len + UInt32(Int(start_pos))
        else:
            var kv_row_offsets = kv_input_row_offsets.value()
            var kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            var kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            var cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = UInt32(cur_kv_len + Int(start_pos))
        q_row = seq_info.start_of_seq

    # Homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_row = batch_idx * max_seq_len.as_uint32()

    var q_offset: Int
    var q_col: UInt32

    comptime if _is_decoding[MaxSeqLenType]():
        # q matrix view is rows x depth
        q_col = 0
        q_offset = depth * Int(
            q_row * UInt32(q_num_heads) + seq_info.head_idx * UInt32(group)
        )
    else:  # head_idx is for q_heads
        # q matrix view is rows x (depth*q_num_heads)
        q_row += seq_info.prompt_offset
        q_col = seq_info.head_idx * UInt32(depth)
        q_offset = depth * q_num_heads * Int(q_row) + Int(q_col)
    ret = {q_row, q_col, q_offset, num_keys, start_pos, seq_info}


@always_inline
def get_q_head_idx[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    //,
](
    position: MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ],
    lane: UInt32,
    out indices: StaticTuple[UInt32, type_of(position).num_q_heads_per_thread],
):
    """Computes the query head indices owned by the current thread.

    In decoding mode each thread holds several grouped query heads spaced by 8,
    while in prefill mode every thread shares the single current head index.

    Parameters:
        BM: Block size in the query-row (M) dimension of the attention tile.
        BN: Block size in the key/value (N) dimension of the attention tile.
        depth: Unpadded head dimension of each attention head.
        padded_depth: Head dimension padded to the MMA alignment boundary.
        num_heads: Number of query heads in the model.
        group: Grouped-query attention group size, query heads per KV head.
        decoding: Whether the kernel runs in single-token decoding mode.

    Args:
        position: Current tile position metadata carrying the head index,
            query row and column offsets, and sequence info.
        lane: Lane index of the calling thread within its warp, used to
            distribute grouped query heads across threads.
    """
    comptime if decoding:
        var q_head_idx_0: UInt32 = UInt32(group) * position.head_idx + lane // 4

        indices = {}
        indices[0] = q_head_idx_0

        comptime for i in range(1, position.num_q_heads_per_thread):
            indices[i] = q_head_idx_0 + UInt32(8 * i)

    else:
        indices = {position.head_idx}


@always_inline
def _apply_mask[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    mask_t: MHAMask,
    reg_tile_layout: Layout,
    element_layout: Layout,
    //,
    # last_iter: Bool,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
](
    mask_warp_row_arg: UInt32,
    position: MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ],
    lane: UInt32,
    max_seq_len: UInt32,
    scale_log2e: Scalar[accum_type],
    kv_tile_start_row: UInt32,
    mask: mask_t,
    mask_status: TileMaskStatus,
    p_reg_tile: LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutAnyOrigin,
        address_space=.LOCAL,
        element_layout=element_layout,
    ],
):
    comptime num_groups_per_thread = min(
        2, ceildiv(group, 8)
    ) if decoding else 2
    var batch_cache_valid_length: UInt32

    comptime if decoding:
        if warp.broadcast(ufloordiv(thread_idx.x - 128, 32)) > (
            (group - 1) // 16
        ):
            return
        if lane >= UInt32(4 * group):
            return
        batch_cache_valid_length = position.num_keys - 1
    else:
        batch_cache_valid_length = 0

    comptime p_frag_simdwidth = element_layout.size()
    # Vectorize by 2.
    var fragment_row: UInt32 = lane // 4
    var fragment_col: UInt32 = (
        lane * UInt32(p_frag_simdwidth) % UInt32(WN)
    ) % 8
    # Offset to current thread's fragment
    var mask_warp_row: UInt32 = mask_warp_row_arg + fragment_row
    var mask_warp_col: UInt32 = kv_tile_start_row + fragment_col

    @always_inline
    def _apply_mask_capture[masked: Bool]() {imm}:
        comptime for m_mma in range(num_m_mmas):
            comptime for n_mma in range(num_n_mmas):
                # Coordinates in mask for current mma tile.
                var mask_frag_row = mask_warp_row + UInt32(m_mma * WM)
                var mask_frag_col = mask_warp_col + UInt32(n_mma * WN)

                comptime for i in range(num_groups_per_thread):
                    var q_head_idx: UInt32 = position.head_idx

                    comptime if decoding:
                        var group_idx = UInt32(i * 8) + fragment_row
                        q_head_idx = UInt32(group) * q_head_idx + group_idx
                    # The row in score matrix of shape seq_len x num_keys.
                    # Mask col is score col since we don't partition in col.
                    var score_row: UInt32
                    var score_row_with_start_pos: UInt32

                    comptime if decoding:
                        score_row = batch_cache_valid_length
                        score_row_with_start_pos = score_row
                    else:
                        score_row = (
                            position.prompt_offset
                            + mask_frag_row
                            + UInt32(i * WM // 2)
                        )
                        score_row_with_start_pos = (
                            score_row + position.start_pos
                        )

                    comptime for j in range(WN // 8):
                        var score_col = mask_frag_col + UInt32(j * 8)
                        var p = p_reg_tile[i, m_mma, j, n_mma]

                        comptime if masked:
                            p = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    Int(position.prompt_idx),
                                    Int(q_head_idx),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p * scale_log2e,
                            )
                        else:
                            p *= scale_log2e

                        comptime if mask_t.apply_log2e_after_mask:
                            p *= log2e

                        var bound: IndexList[2, element_type=.uint32]

                        comptime if decoding:
                            bound = IndexList[2, element_type=.uint32](
                                Int(position.num_keys),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type=.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        elif masked:
                            bound = IndexList[2, element_type=.uint32](
                                Int(position.seq_len),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type=.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        p_reg_tile[i, m_mma, j, n_mma] = p

    comptime if decoding:
        _apply_mask_capture[True]()
    else:
        unswitch(
            (mask_status == TileMaskStatus.PARTIAL_MASK)
            # NOTE: mask_status should be either PARTIAL_MASK or NO_MASK at
            # this point.
            # In the NO_MASK case, we still need to mask out the scores for the
            # last tile, which goes beyond num_keys (for num_keys % 128 != 0).
            or (UInt32(BN) + kv_tile_start_row > position.num_keys),
            _apply_mask_capture,
        )


@always_inline
def produce[
    qkv_type: DType,
    BM: Int,
    BN: Int,
    q_rank: Int,
    q_tile_shape: IndexList[q_rank],
    q_desc_shape: IndexList[q_rank],
    depth: Int,
    padded_depth: Int,
    num_heads: Int,
    group: Int,
    PartitionType: MHAPartitionScheme,
    MaxSeqLenType: OptionallyStaticInt,
    SchedulerType: MHATileScheduler,
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    KVInputRowOffsetsType: OptionalPointer,
    ValidLengthType: OptionalPointer,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    pipeline_stages: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    q_tma_op: TMATensorTile[
        qkv_type,
        q_rank,
        q_tile_shape,
        q_desc_shape,
    ],
    k_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_tile_rows(BN, KVLUTType.page_size),
        BK=padded_depth,
    ],
    v_tma_op: KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_tile_rows(BN, KVLUTType.page_size),
        BK=padded_depth,
    ],
    q_smem: UnsafePointer[mut=True, Scalar[qkv_type], _, address_space=.SHARED],
    kv_smem: UnsafePointer[
        mut=True, Scalar[qkv_type], _, address_space=.SHARED
    ],
    produced_mbar_kv: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    consumed_mbar_kv: UnsafePointer[
        mut=True, SharedMemBarrier, _, address_space=.SHARED
    ],
    produced_mbar_q: Optional[
        UnsafePointer[SharedMemBarrier, MutAnyOrigin, address_space=.SHARED]
    ],
    consumed_mbar_q: Optional[
        UnsafePointer[SharedMemBarrier, MutAnyOrigin, address_space=.SHARED]
    ],
    kv_lut: KVLUTType,
    initial_position: MHAPosition[
        BM,
        BN,
        depth,
        padded_depth,
        num_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ],
    partition: PartitionType,
    scheduler: SchedulerType,
    mask: MaskType,
    tile_summary: MHATileSummary[ValidLengthType],
    tile_state_arg: MHATileState,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    kv_input_row_offsets: KVInputRowOffsetsType,
):
    """Stages Q, K, and V tiles into shared memory via TMA for the SM90 attention kernel.

    Drives the producer side of the warp-specialized pipeline: it issues the
    preheader Q and K copies, then runs the main loop emitting K and V copies
    (full or partial-page) synchronized through the shared-memory barriers,
    advancing the persistent scheduler across tiles when configured.

    Parameters:
        qkv_type: Element dtype of the query, key, and value tensors
            (inferred).
        BM: Block size in the query-row (M) dimension of the attention
            tile (inferred).
        BN: Block size in the key/value (N) dimension of the attention
            tile (inferred).
        q_rank: Rank of the query TMA tensor map descriptor (inferred).
        q_tile_shape: Per-dimension tile shape for query TMA copies
            (inferred).
        q_desc_shape: Per-dimension descriptor shape of the query TMA
            tensor map (inferred).
        depth: Unpadded head dimension of each attention head (inferred).
        padded_depth: Head dimension padded to the MMA alignment boundary
            (inferred).
        num_heads: Number of query heads in the model (inferred).
        group: Grouped-query attention group size, query heads per KV head
            (inferred).
        PartitionType: Scheme dividing the KV range across producer
            instances (inferred).
        MaxSeqLenType: Statically or dynamically known maximum sequence
            length (inferred).
        SchedulerType: Tile scheduler driving persistent tile iteration
            (inferred).
        KVLUTType: KV cache operand type providing paged row lookup and
            cache length queries (inferred).
        MaskType: Attention mask type applied to score tiles (inferred).
        KVInputRowOffsetsType: Optional pointer type for cross-attention
            KV row offsets, null when unused (inferred).
        ValidLengthType: Optional pointer type for per-batch valid lengths
            passed to the tile scheduler (inferred).
        swizzle_mode: Shared-memory swizzle mode applied to TMA tensor
            maps.
        pipeline_stages: Number of double-buffered pipeline stages for KV
            copies, must be at least 2.
        ragged: Whether ragged (variable-length) batching is in use.
        _is_cache_length_accurate: Whether the KV cache length is already
            accurate, skipping the per-batch cache_length lookup when
            true.

    Args:
        q_tma_op: TMA tile descriptor issuing asynchronous query tile copies
            into shared memory.
        k_tma_op: TMA tile descriptor issuing asynchronous key tile copies
            into shared memory.
        v_tma_op: TMA tile descriptor issuing asynchronous value tile copies
            into shared memory.
        q_smem: Base shared-memory pointer for query tile staging.
        kv_smem: Base shared-memory pointer for key and value tile staging.
        produced_mbar_kv: Shared-memory barriers signaling KV tile
            production completion to the consumer.
        consumed_mbar_kv: Shared-memory barriers waited on before
            overwriting a KV pipeline stage.
        produced_mbar_q: Optional barriers signaling query tile production
            for persistent kernels, unused otherwise.
        consumed_mbar_q: Optional barriers waited on before overwriting a
            query pipeline stage in persistent kernels.
        kv_lut: KV cache operand providing paged row lookup and cache
            length queries.
        initial_position: Starting tile position metadata for the first
            query and key tiles.
        partition: Partition instance dividing the KV range for this
            producer.
        scheduler: Tile scheduler instance advancing persistent tile
            iteration.
        mask: Attention mask instance applied to score tiles.
        tile_summary: Tile summary carrying per-batch valid lengths for
            the scheduler.
        tile_state_arg: Initial scheduler state, mutated as tiles advance.
        max_seq_len: Maximum sequence length after padding, used for
            homogeneous batch query row offsets.
        num_keys_arg: Number of cache keys for the current batch, used in
            homogeneous batching.
        kv_input_row_offsets: Optional per-batch KV row offsets for cross
            attention, null when unused.
    """
    comptime swizzle_granularity = swizzle_mode.bytes() // size_of[qkv_type]()

    comptime decoding: Bool = _is_decoding[MaxSeqLenType]()
    comptime PositionType = MHAPosition[
        BM, BN, depth, padded_depth, num_heads, group, decoding
    ]
    comptime persistent = SchedulerType.may_advance

    comptime q_smem_layout_consumer = tile_layout_k_major[
        qkv_type, BM, padded_depth, swizzle_mode=swizzle_mode
    ]()

    comptime q_size = q_smem_layout_consumer.size()
    comptime q_smem_size = (2 * q_size if persistent else q_size)

    comptime q_copy_rows = max(group, 8) if decoding else BM
    comptime qk_bytes = (q_copy_rows + BN) * padded_depth * size_of[qkv_type]()

    var tile_state = tile_state_arg
    var position = initial_position

    @__parameter
    @always_inline("nodebug")
    def q_producer(
        q_idx: UInt32, offset: UInt32 = 0
    ) -> LayoutTensor[
        qkv_type,
        Layout.row_major(q_tile_shape),
        type_of(q_smem).origin,
        address_space=.SHARED,
        alignment=128,
    ]:
        return {q_smem + UInt32(q_size) * q_idx + offset}

    comptime k_smem_layout = tile_layout_k_major[
        qkv_type, BN, padded_depth, swizzle_mode
    ]()
    comptime assert pipeline_stages >= 2

    @__parameter
    @always_inline
    def kv_tile(
        idx: UInt32,
        out tile: LayoutTensor[
            qkv_type,
            k_smem_layout,
            type_of(kv_smem).origin,
            address_space=.SHARED,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            alignment=128,
        ],
    ):
        comptime sz = BN * padded_depth
        tile = {kv_smem + UInt32(sz) * idx}

    comptime kv_sub_BN = kv_sub_tile_rows(BN, KVLUTType.page_size)

    comptime KVTMA = KVTMATile[
        qkv_type,
        swizzle_mode,
        BN=kv_sub_BN,
        BK=padded_depth,
    ]
    comptime KVPagedRows = PagedRowIndices[BN, KVLUTType.page_size]
    comptime page_size = KVLUTType.page_size
    comptime needs_partial = page_size > 0 and page_size < BN
    comptime kv_bytes_pp = padded_depth * KVPagedRows.eff_page * size_of[
        qkv_type
    ]()
    # Alignment of `kv_tile_row` produced by mask-driven iteration.
    comptime base_alignment: Int = MaskType.start_column_alignment[
        BM, BN, page_size
    ]()

    @__parameter
    @always_inline("nodebug")
    def _num_valid_pages(end_row: UInt32, current_kv_row: UInt32) -> UInt32:
        return min(
            UInt32(ceildiv(Int(end_row - current_kv_row), page_size)),
            UInt32(KVPagedRows.num_pages),
        )

    @__parameter
    @always_inline("nodebug")
    def produce_kv[
        is_k_side: Bool,
        wait: Bool,
    ](
        tma_op: KVTMA,
        mut state: PipelineState[pipeline_stages],
        prompt_idx: UInt32,
        kv_tile_row: UInt32,
        kv_head_idx: UInt32,
    ):
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]

        comptime if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            comptime bytes = BN * padded_depth * size_of[qkv_type]()
            p_mbar.expect_bytes(Int32(bytes))

        comptime stage_sz = BN * padded_depth
        var paged_rows = kv_lut.populate[BN, base_alignment](
            prompt_idx, kv_tile_row
        )
        # SM90's producer runs only on thread 0 (gated by `if thread_idx.x == 0:`
        # in `sm90/mha.mojo`), so `elect()` elects that lone thread and returns 1.
        # Forward it instead of a hard-coded `1` so the TMA's elect predicate
        # stays `elect.sync`-derived (and self-protecting if the gate is widened).
        var e = elect()
        comptime if is_k_side:
            paged_rows.tma_copy_k[needs_partial=False](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=e,
            )
        else:
            paged_rows.tma_copy_v[needs_partial=False](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=e,
            )
        state.step()

    @__parameter
    @always_inline("nodebug")
    def produce_kv_partial[
        is_k_side: Bool,
        wait: Bool,
    ](
        tma_op: KVTMA,
        mut state: PipelineState[pipeline_stages],
        prompt_idx: UInt32,
        kv_tile_row: UInt32,
        kv_head_idx: UInt32,
        num_valid_pages: UInt32,
    ):
        """Like `produce_kv` but with runtime-bounded page loops.

        Uses `populate` + `tma_copy_{k,v}[needs_partial=True]` so the
        row-index lookup (one SIMD LUT load for the paged case) is
        amortized across all TMA copies issued by this call.
        """
        var write_idx: UInt32 = state.index()
        var write_phase: UInt32 = state.phase()

        ref p_mbar = produced_mbar_kv[write_idx]

        comptime if wait:
            consumed_mbar_kv[write_idx].wait(write_phase)
            p_mbar.expect_bytes(Int32(kv_bytes_pp * Int(num_valid_pages)))

        comptime stage_sz = BN * padded_depth
        # See `produce_kv`: SM90 producer is single-threaded, so `elect()` elects
        # the lone thread (returns 1); forward it to keep the predicate
        # `elect.sync`-derived.
        var paged_rows = kv_lut.populate[BN, base_alignment](
            prompt_idx, kv_tile_row
        )
        var e = elect()
        comptime if is_k_side:
            paged_rows.tma_copy_k[needs_partial=True](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=e,
                k_num_valid_pages=num_valid_pages,
            )
        else:
            paged_rows.tma_copy_v[needs_partial=True](
                tma_op,
                kv_smem + UInt32(stage_sz) * write_idx,
                p_mbar,
                kv_head_idx=kv_head_idx,
                elect=e,
                num_valid_pages=num_valid_pages,
            )
        state.step()

    @__parameter
    @always_inline
    def get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM,
            BN,
            depth,
            padded_depth,
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

    var write_pipeline_states = PipelineState[pipeline_stages]()
    var q_pipeline_state = PipelineState[2 if persistent else 1]()

    var start: UInt32
    var end: UInt32
    comptime if PartitionType.do_partition:
        var startend = position.get_start_and_end_for_partitions[
            page_size=KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
        if start >= end:
            return
    else:
        # delay partitioning until after we've begun copying `q`
        start = 0
        end = 0

    comptime q_bytes = q_copy_rows * padded_depth * size_of[qkv_type]()
    comptime if not needs_partial:
        produced_mbar_kv[0].expect_bytes(Int32(qk_bytes))

    comptime if decoding:
        ref q_mbar = produced_mbar_kv[0]
        var q_idx: UInt32 = q_pipeline_state.index()

        comptime for d_idx in range(ceildiv(depth, swizzle_granularity)):
            comptime d: Int = d_idx * swizzle_granularity
            comptime smem_offset = q_smem_layout_consumer(IntTuple(0, d))

            q_tma_op.async_copy_4d(
                q_producer(q_idx, UInt32(smem_offset)),
                q_mbar,
                (
                    d,
                    0,
                    Int(position.head_idx),
                    Int(position.q_row),
                ),
            )
    else:
        q_tma_op.async_copy(
            q_producer(q_pipeline_state.index()),
            produced_mbar_kv[0],
            q_coord[
                depth=depth,
                decoding=decoding,
            ](position.q_row, position.head_idx),
        )

    comptime if not PartitionType.do_partition:
        var startend = position.get_start_and_end_for_partitions[
            page_size=KVLUTType.page_size
        ](partition, mask)
        start = startend[0]
        end = startend[1]
    var kv_tile_start_row: UInt32 = start

    while (
        position.mask_status(mask, kv_tile_start_row)
        == TileMaskStatus.FULL_MASK
    ):
        kv_tile_start_row += UInt32(BN)

    var kv_head_idx: UInt32 = position.kv_head_idx()

    # When needs_partial, nvp0 tracks the valid page count for K0.
    # Initialize to num_pages (full) and overwrite inside the comptime if.
    var nvp0: UInt32 = UInt32(KVPagedRows.num_pages)
    comptime if needs_partial:
        nvp0 = _num_valid_pages(end, kv_tile_start_row)
        produced_mbar_kv[0].expect_bytes(
            Int32(q_bytes + kv_bytes_pp * Int(nvp0))
        )
        if nvp0 < UInt32(KVPagedRows.num_pages):
            produce_kv_partial[is_k_side=True, wait=False](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
                nvp0,
            )
        else:
            produce_kv[is_k_side=True, wait=False](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
            )
    else:
        produce_kv[is_k_side=True, wait=False](
            k_tma_op,
            write_pipeline_states,
            position.prompt_idx,
            kv_tile_start_row,
            kv_head_idx,
        )

    var kv_head_idx_prev: UInt32 = kv_head_idx
    var kv_tile_start_row_prev: UInt32 = kv_tile_start_row
    var nvp_prev: UInt32 = nvp0
    # wait to flip phase, but only bother after producing
    # there isn't any memory we can throttle
    # the order of the consumer's arrivals determines the
    # order of the producer's waits.
    # few_keys = num_keys <= BN

    var new_end: UInt32
    # Process work with the tile size until there's not enough remaining work
    # to fit in a tile.
    # Production order:
    # Preheader: Q0, K0
    # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
    # Exit: V{-1}
    while True:
        # this loops over num_keys
        kv_tile_start_row += UInt32(BN)
        if kv_tile_start_row >= end:
            comptime if persistent:
                kv_tile_start_row = 0
                var q_idx_old: UInt32 = q_pipeline_state.index()
                var q_phase_old: UInt32 = q_pipeline_state.phase()
                q_pipeline_state.step()
                consumed_mbar_q.unsafe_value()[q_idx_old].wait(q_phase_old)
                # we must wait before advancing, as this mbar
                # is for both `q_smem` and `sidx_ptr`
                var q_idx: UInt32 = q_pipeline_state.index()
                var docontinue = scheduler.advance[
                    producer=True, sync=MHASchedulerSynchronization.DEFAULT
                ](tile_summary, tile_state, q_idx_old)
                # FIXME: persistent kernel that uses a counter
                # must signal somehow
                if not docontinue:
                    break
                ref pq_mbar = produced_mbar_q.unsafe_value()[q_idx_old]
                position = get_position(docontinue.value())
                pq_mbar.expect_bytes(
                    Int32(q_copy_rows * padded_depth * size_of[qkv_type]())
                )

                comptime if not decoding:
                    q_tma_op.async_copy(
                        q_producer(q_idx),
                        pq_mbar,
                        q_coord[
                            depth=depth,
                            decoding=decoding,
                        ](position.q_row, position.head_idx),
                    )

                else:
                    comptime for d_idx in range(depth // 64):
                        comptime d: Int = d_idx * 64
                        q_tma_op.async_copy_4d(
                            q_producer(q_idx),
                            pq_mbar,
                            (
                                d,
                                0,
                                Int(position.head_idx),
                                Int(position.q_row),
                            ),
                        )

                kv_head_idx = position.kv_head_idx()
                start, new_end = position.get_start_and_end_for_partitions[
                    page_size=KVLUTType.page_size
                ](partition, mask)
                kv_tile_start_row = start
                end = new_end
            else:
                break

        if (
            position.mask_status(mask, kv_tile_start_row)
            == TileMaskStatus.FULL_MASK
        ):
            continue

        comptime if needs_partial:
            # Compute valid page count for current K tile.
            var nvp_cur: UInt32
            if kv_tile_start_row + UInt32(BN) > end:
                nvp_cur = _num_valid_pages(end, kv_tile_start_row)
            else:
                nvp_cur = UInt32(KVPagedRows.num_pages)

            # K: use partial if this tile doesn't fill all pages.
            if nvp_cur < UInt32(KVPagedRows.num_pages):
                produce_kv_partial[is_k_side=True, wait=True](
                    k_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row,
                    kv_head_idx,
                    nvp_cur,
                )
            else:
                produce_kv[is_k_side=True, wait=True](
                    k_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row,
                    kv_head_idx,
                )

            # V: for previous K's tile. Use partial if that K was partial.
            if nvp_prev < UInt32(KVPagedRows.num_pages):
                produce_kv_partial[is_k_side=False, wait=True](
                    v_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row_prev,
                    kv_head_idx_prev,
                    nvp_prev,
                )
            else:
                produce_kv[is_k_side=False, wait=True](
                    v_tma_op,
                    write_pipeline_states,
                    position.prompt_idx,
                    kv_tile_start_row_prev,
                    kv_head_idx_prev,
                )

            nvp_prev = nvp_cur
        else:
            produce_kv[is_k_side=True, wait=True](
                k_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row,
                kv_head_idx,
            )
            produce_kv[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
            )
        kv_head_idx_prev = kv_head_idx
        kv_tile_start_row_prev = kv_tile_start_row

    # Exit: V for the last K tile.
    comptime if needs_partial:
        if nvp_prev < UInt32(KVPagedRows.num_pages):
            produce_kv_partial[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
                nvp_prev,
            )
        else:
            produce_kv[is_k_side=False, wait=True](
                v_tma_op,
                write_pipeline_states,
                position.prompt_idx,
                kv_tile_start_row_prev,
                kv_head_idx_prev,
            )
    else:
        produce_kv[is_k_side=False, wait=True](
            v_tma_op,
            write_pipeline_states,
            position.prompt_idx,
            kv_tile_start_row_prev,
            kv_head_idx_prev,
        )


@always_inline
@always_inline
def output_reg_to_smem[
    output_type: DType,
    accum_type: DType,
    num_m_mmas: Int,
    o_frag_size: Int,
    //,
    BM: Int,
    BN: Int,
    padded_depth: Int,
    swizzle: Swizzle,
    num_consumer: Int,
](
    tid: UInt32,
    local_warp_group_idx: UInt32,
    warp_y: UInt32,
    q_smem: UnsafePointer[
        Scalar[output_type], MutAnyOrigin, address_space=.SHARED
    ],
    output_reg_tile: LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas, o_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ],
) -> LayoutTensor[
    output_type,
    Layout.row_major(BM, padded_depth),
    MutAnyOrigin,
    address_space=.SHARED,
]:
    """Stores the output accumulator registers from local memory into shared memory.

    Selects the `stmatrix`-based path for float32 accumulators with half-precision
    output and aligned depths, otherwise falls back to a per-thread
    `copy_local_to_shared` store.

    Parameters:
        output_type: DType of the values written to shared memory.
        accum_type: DType of the accumulator values held in registers.
        num_m_mmas: Number of MMA tiles in the M dimension of the output
            register tile.
        o_frag_size: Number of elements per output fragment per MMA tile.
        BM: Block size in the query-row (M) dimension of the output tile.
        BN: Block size in the key/value (N) dimension of the output warp tile.
        padded_depth: Head dimension padded to the MMA alignment boundary.
        swizzle: Shared-memory swizzle pattern applied to the output store
            layout.
        num_consumer: Number of consumer warp groups writing the output
            concurrently.

    Args:
        tid: Thread index of the calling thread, used to derive the warp-group
            thread index for the stmatrix path.
        local_warp_group_idx: Index of the calling warp group among the
            consumer warp groups.
        warp_y: Row index of the calling warp's 16-row tile within the
            BM-row output block.
        q_smem: Base pointer to shared memory where the output tile is stored.
        output_reg_tile: Local register tile holding the accumulator output
            fragments.
    """
    var accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM, padded_depth),
        address_space=.SHARED,
    ](q_smem)
    comptime use_stmatrix = accum_type == DType.float32 and padded_depth % 16 == 0 and size_of[
        output_type
    ]() == 2 and o_frag_size % 8 == 0

    comptime if use_stmatrix:
        var warp_group_thread_idx = tid % UInt32(WARPGROUP_SIZE)
        comptime reg_layout = row_major[num_m_mmas, o_frag_size]()
        comptime smem_layout = row_major[BM, padded_depth]()
        output_reg_to_smem_st_matrix[BM, swizzle, num_consumer](
            warp_group_thread_idx,
            local_warp_group_idx,
            _LocalTT[accum_type, reg_layout](output_reg_tile.ptr, reg_layout),
            _SharedMemTT[output_type, smem_layout](
                accum_smem_tile.ptr, smem_layout
            ),
        )
    else:
        comptime mma_thread_layout = Layout.row_major(8, 4)
        var accum_smem_warp_tile = accum_smem_tile.tile[16, BN](
            Int(warp_y), Int(0)
        )
        copy_local_to_shared[thread_layout=mma_thread_layout, swizzle=swizzle](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )
    return accum_smem_tile
