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

"""Provides shared utilities for SM100 MLA decode attention kernels.

Defines TMA tile helpers, pipeline producer/consumer structs, MMA tensor
accumulator descriptors, and the common softmax/correction/store logic reused
across the BF16, native FP8, and per-token-scale decode backends.
"""

from std.collections import OptionalReg
from std.math import exp2, recip, align_up, log2, ceildiv
from std.math.constants import log2e
from std.sys import size_of, _RegisterPackType
from max.gpu import thread_idx, block_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from max.gpu.host.info import B200
from max.gpu.memory import fence_async_view_proxy
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
)

from max.gpu.sync import named_barrier
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_fence_after,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
    tcgen05_store_wait,
)
from max.gpu.primitives.warp import _vote_nvidia_helper
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from layout.tile_layout import row_major as tt_row_major
from layout.swizzle import make_ldmatrix_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import (
    create_tensor_tile,
    PipelineState,
    _default_desc_shape,
    TMATensorTile,
)
from std.memory import bitcast
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask, MASK_VALUE
from nn.attention.mha_operand import MHAOperand
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor

from nn.attention.gpu.nvidia.sm100.attention import SM100_RESERVED_SMEM_BYTES
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    LocalTensor,
    SharedMemPointer,
    MBarType,
    elect_mma_arrive,
    ProducerPipeline,
    ConsumerPipeline,
    MBarPipeline,
    sub_ftz,
    bulk_mma_ws,
    bulk_mma_ws_ts,
    st_shared_v4_b32,
)
from nn.attention.gpu.nvidia.common import KVTMATile
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.sys._assembly import inlined_assembly


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding TMA tiles
# ------------------------------------------------------------------------------

comptime QOTMATile[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = TMATensorTile[
    dtype,
    2,
    IndexList[2](BM, BK),
    _default_desc_shape[2, dtype, IndexList[2](BM, BK), swizzle_mode](),
    is_k_major=True,
]


@always_inline
def tma_tile_qo[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BK: Int,
    depth: Int,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    rows: Int,
    out res: QOTMATile[dtype, BM, BK, swizzle_mode],
) raises:
    """Creates a TMA descriptor for the Q or output tensor used in MLA decode.

    Parameters:
        dtype: Element type of the Q or output tensor (inferred).
        swizzle_mode: TMA swizzle mode applied to the descriptor.
        BM: Tile height in rows for each TMA copy.
        BK: Tile width in columns for each TMA copy.
        depth: Column count of the full Q or output tensor.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the Q or output tensor in device memory.
        rows: Number of rows in the full Q or output tensor.
    """
    comptime layout = Layout.row_major(UNKNOWN_VALUE, depth)
    var rt_layout = RuntimeLayout[layout].row_major(IndexList[2](rows, depth))
    var tensor = LayoutTensor[dtype, layout](ptr, rt_layout)

    res = rebind[QOTMATile[dtype, BM, BK, swizzle_mode]](
        create_tensor_tile[
            IndexList[2](BM, BK),
            swizzle_mode=swizzle_mode,
        ](ctx, tensor)
    )


# Output TMA tile whose stored row count is chosen per copy. The row axis is
# its own descriptor dim of extent BM, so a row coordinate of
# BM - rows_to_store leaves rows_to_store box rows in bounds and the TMA
# masks the rest. tma_tile_o builds the descriptor and store_row_coords the
# coordinate.
# Similar ragged masking technique as RaggedTMA3DTile in
# layout/tma_async.mojo; consider merging the two in the future.
comptime ORaggedTMATile[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = TMATensorTile[
    dtype,
    3,
    IndexList[3](1, BM, BK),
    _default_desc_shape[3, dtype, IndexList[3](1, BM, BK), swizzle_mode](),
    is_k_major=True,
]


@always_inline
def tma_tile_o[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BK: Int,
    depth: Int,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    rows: Int,
    out res: ORaggedTMATile[dtype, BM, BK, swizzle_mode],
) raises:
    """Creates the MLA decode output TMA descriptor.

    The row axis is its own descriptor dim of extent BM, reached through a
    separate outer dim with the same row stride. This makes the stored row
    count a per-copy coordinate, built by store_row_coords. Box row r lands
    on tensor row row + r, and box rows at or past rows_to_store fall
    outside the extent and are dropped. The descriptor base sits BM rows
    before ptr, but masked rows are never written, so memory before ptr is
    never touched.

    Parameters:
        dtype: Element type of the output tensor (inferred).
        swizzle_mode: TMA swizzle mode applied to the descriptor.
        BM: Row extent of the box.
        BK: Tile width in columns for each TMA copy.
        depth: Column count of the full output tensor.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the output tensor in device memory.
        rows: Number of rows in the full output tensor.
    """
    # Outer coordinates run up to rows, so the extent is one past that. Its
    # box is 1, so it never masks.
    res = create_tma_descriptor[dtype, 3, swizzle_mode](
        DeviceBuffer(ctx, ptr - depth * BM, 1, owning=False),
        IndexList[3](rows + 1, BM, depth),
        IndexList[3](depth, depth, 1),
        _default_desc_shape[3, dtype, IndexList[3](1, BM, BK), swizzle_mode](),
    )


@always_inline
def store_row_coords[
    BM: Int
](col: Int, row: Int, rows_to_store: Int) -> Tuple[Int, Int, Int]:
    """Builds the ORaggedTMATile store coordinate for a partial row count.

    Args:
        col: Column offset within the output row.
        row: First output tensor row the copy writes.
        rows_to_store: How many rows of the box are real, at most BM.

    Returns:
        Coordinates for async_store_3d, ordered innermost dim first.
    """
    return (col, BM - rows_to_store, row + rows_to_store)


# Per-token scales TMA tile: loads BN_QK contiguous float32 values via TMA.
# Scales are treated as a flat 1D array indexed by row_idx (same paging
# as the KV cache blocks). The TMA uses a [1, total_elements] 2D layout
# so the inner dimension (total_elements * 4 bytes) exceeds the TMA minimum
# of 32 bytes, with tile shape [1, BN_QK] and SWIZZLE_NONE.
#
# We set desc_shape = tile_shape (no sub-tiling) so that desc_bytes ==
# tile_bytes and the 128-byte alignment constraint for multi-copy TMA is
# not triggered. With BN_QK=64, tile_bytes = 256 which is already 128-aligned.
comptime ScalesTMATile[BN_QK: Int] = TMATensorTile[
    DType.float32,
    2,
    IndexList[2](1, BN_QK),
    IndexList[2](1, BN_QK),
    is_k_major=True,
]


@always_inline
def tma_tile_scales[
    BN_QK: Int,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Float32, origin=MutAnyOrigin],
    total_elements: Int,
    out res: ScalesTMATile[BN_QK],
) raises:
    """Create a TMA descriptor for per-token float32 scales.

    The scales are a flat array of float32 values indexed by the same
    row_idx as the KV cache blocks. We create a 2D TMA with shape
    [1, total_elements] and tile [1, BN_QK] so that each async_copy loads
    BN_QK contiguous float32 values (BN_QK * 4 bytes) starting at the
    specified column offset.

    Parameters:
        BN_QK: Tile width in float32 values loaded by each async_copy,
            equal to the KV cache tile width.

    Args:
        ctx: Device context used to create the TMA descriptor.
        ptr: Base pointer of the flat float32 scales array in device
            memory, indexed by the same row index as the KV cache blocks.
        total_elements: Total number of float32 scales in the array,
            used as the inner (column) dimension of the 2D TMA descriptor.
    """
    comptime layout = Layout.row_major(1, UNKNOWN_VALUE)
    var rt_layout = RuntimeLayout[layout].row_major(
        IndexList[2](1, total_elements)
    )
    var tensor = LayoutTensor[.float32, layout, MutAnyOrigin](ptr, rt_layout)
    res = rebind[ScalesTMATile[BN_QK]](
        create_tensor_tile[
            IndexList[2](1, BN_QK),
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            __desc_shape=IndexList[2](1, BN_QK),
        ](ctx, tensor)
    )


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding pack
# ------------------------------------------------------------------------------


struct MLA_Decode_Pack[
    ValidLengthType: OptionalPointer,
    MaskType: MHAMask,
    SplitAccumType: OptionalPointer,
](Copyable, DevicePassable, TrivialRegisterPassable):
    """Bundles the mask, valid-length, and split-K accumulator pointers passed to decode kernels.

    Parameters:
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor (may be `Null` when unused).
        MaskType: `MHAMask` type applied to the attention scores.
        SplitAccumType: `OptionalPointer` type wrapping the split-K LSE
            accumulator buffer (may be `Null` when split-K is unused).
    """

    var mask: Self.MaskType
    var valid_length: Self.ValidLengthType
    var lse_accum_split_ptr: Self.SplitAccumType
    var num_partitions: Int
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "Pack"

    @staticmethod
    def get_device_type_name() -> String:
        return Self.get_type_name()

    @always_inline
    def __init__(
        out self,
        mask: Self.MaskType,
        valid_length: Self.ValidLengthType,
        lse_accum_split_ptr: Self.SplitAccumType,
        num_partitions: Int,
    ):
        self.mask = mask
        self.valid_length = valid_length
        self.lse_accum_split_ptr = lse_accum_split_ptr
        self.num_partitions = num_partitions


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------


@always_inline
def num_matrix_view_rows_decode[
    dtype: DType,
    //,
](q: TileTensor[dtype, ...]) -> Int:
    """TileTensor overload of `num_matrix_view_rows_decode`.

    Parameters:
        dtype: Element type of the query tile tensor (inferred).

    Args:
        q: Query tile tensor whose leading `rank - 1` dimensions are
            multiplied to compute the total matrix view row count.
    """
    # q and output are (batch x seq_len x num_heads , depth)
    # output when split-k is used are (split_k x batch x seq_len x num_heads , depth)
    var num_rows = Int(q.dim[0]())

    comptime for i in range(1, q.rank - 1):
        num_rows *= Int(q.dim[i]())
    return num_rows


# SharedMemPointer and MBarType are imported from
# nn.attention.gpu.nvidia.sm100.attention_utils (centralized shared memory type aliases).


# ------------------------------------------------------------------------------
# MLA decoding configuration for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Config:
    """Holds the tile sizes, swizzle modes, and SMEM/TMEM layout for an SM100 MLA decode kernel.
    """

    var MMA_M: Int
    var MMA_PV_N: Int
    var MMA_QK_N: Int
    var BM: Int
    var BN_PV: Int  # N of PV MMA = output (V) head_dim per CTA. Anchors output writeback path.
    var BN_QK: Int  # N of QK MMA = KV cache tile width (keys per k-tile)
    # K of the QK MMA. The SMEM tile width, not the gmem row width (see
    # `input_q_depth`), so the NoPE tail stays contracted as zero.
    var BK_QK: Int
    var q_depth: Int
    var depth: Int  # this is V depth
    var padded_depth: Int
    var padded_q_depth: Int
    # Gmem row width the TMA reads. May be narrower than `padded_q_depth` for a
    # NoPE model, which zero-fills the tail in SMEM. Drives the TMA descriptors
    # and byte count only, never the SMEM layout or MMA width.
    var input_q_depth: Int
    var rope_depth: Int  # this is Q depth - V depth
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_O: Int = 0
    comptime TMEM_S0: Int = Self.TMEM_O + 256
    comptime TMEM_S1: Int = Self.TMEM_S0 + 32
    # Upper bound on pipeline stages — kernels pick `num_kv_stages` ≤ this
    # at config time, and TMEM is laid out to fit the worst case so
    # downstream offsets (CORR_SCALE etc.) stay stable across configs.
    # 6 stages × 32 cols/stage = 192 cols, sitting in TMEM cols 256..447
    # between TMEM_O (256 cols) and CORR_SCALE (col 448), within the 512
    # total TMEM columns on SM100. Bump only if a kernel needs deeper
    # KV pipelining and the resulting layout still fits.
    comptime MAX_TMEM_S_SLOTS: Int = 6
    comptime TMEM_CORR_SCALE: Int = Self.TMEM_S0 + Self.MAX_TMEM_S_SLOTS * 32
    comptime TMEM_CORR_LI: Int = Self.TMEM_CORR_SCALE + 1
    var tmem_used: Int
    var num_kv_stages: Int
    # Depth of the MMA-operand KV ring. Equal to `num_kv_stages` everywhere
    # except the unified-gather path, whose two KV buffers need not share a
    # depth: only the one the MMA consumes carries the loop-carried WAR edge
    # that bounds the steady-state rate.
    var num_kv_mma_stages: Int
    var smem_used: Int
    var dtype_size: Int
    var num_threads: Int  # bf16: 3 WGs (MMA, softmax, correction); fp8: 4 WGs (+convert)
    var swizzle_mode: TensorMapSwizzle
    var kv_mma_swizzle_mode: TensorMapSwizzle
    var kv_tma_swizzle_mode: TensorMapSwizzle
    var content_swizzle_mode: TensorMapSwizzle  # FP8 content: SWIZZLE_64B
    var rope_swizzle_mode: TensorMapSwizzle  # BF16 rope: SWIZZLE_128B
    comptime MMA_K = 16
    comptime sm100_smem_carveout = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()  # 8
    comptime cta_group = 1  # TODO: support 2
    var decoding_warp_split_k: Bool
    var out_rows: Int
    var page_size: Int  # KV cache physical page size (e.g., 128)
    var split_page_size: Int  # Page size for split-K work partitioning (must be <= page_size)
    var scale_block_size: Int  # 0 = tensorwise, 32/64/128 = blockwise FP8 scaling
    var scales_per_token: Int  # ceildiv(q_depth, scale_block_size) when blockwise, else 0
    var scale_smem_per_stage: Int  # BN_QK * scales_per_token bytes per stage (0 if tensorwise)
    var per_token_scale_rope_aware: Bool  # Split content(FP8)/rope(BF16) with per-token FP8 scaling
    var per_token_scales_per_stage: Int  # BN_QK(64) tokens * 1 scale * sizeof(float32)(4) = 256 bytes per stage
    var decode_layout_g: Bool  # Layout G fold path (BM=32, MMA_M=32)
    var BK_PV: Int  # K of PV MMA. Sentinel default = BN_QK.
    # Threshold (in log2 domain) below which the softmax rescale of the
    # running output `O` and `mi` is SKIPPED. When `diff = mi - new_max`
    # is greater than or equal to this threshold (i.e. the running max
    # grew by less than `-threshold` in log2 domain), the rescale is
    # treated as a no-op (scale_for_old_max = 1.0, new_max = mi). This
    # matches FlashMLA's `-6.0` (TokenSpeed's `6.0`) and trades a tiny
    # numerical bias for a saved `exp2` + TMEM publish on every iteration
    # where the running max grows by less than ~6 in log2 domain.
    var skip_correction_threshold: Float32

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        q_depth: Int,
        dtype_size: Int,
        kv_type_size: Int,
        swizzle_mode: TensorMapSwizzle,
        kv_mma_swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        decoding_warp_split_k: Bool,
        split_page_size: Int = 128,
        scale_block_size: Int = 0,
        native_fp8: Bool = False,
        per_token_scale_rope_aware: Bool = False,
        # Selects the Layout G fork of the qkv_fp8 native-FP8 kernel
        # (1x4 datapath, BM=32, MMA_M=32). False is the default for all
        # other backends.
        decode_layout_g: Bool = False,
        # Selects the unified native-FP8 sparse decode kernel
        # (MLA_SM100_Decode_Sparse_QKV_FP8's re-swizzle-staged gather):
        # KV is gathered contiguously (SWIZZLE_NONE/int64, matching the old
        # converter kernel's efficient gather) into a LINEAR staging buffer,
        # then a re-swizzle warpgroup permutes it FP8->FP8 into the SW64
        # layout the native tcgen05.mma.kind::f8f6f4 operand expects. Adds a
        # second, same-sized KV buffer (linear-in) + a second KV pipeline's
        # worth of barriers per stage, and goes back to 4 warpgroups (the
        # re-swizzle WG). False (default) preserves the plain native-FP8
        # single-buffer SW64-gather layout for every other call site
        # (dense native FP8, Layout G).
        native_fp8_unified_gather: Bool = False,
        # Sentinel `0` means "use the default (64)" for decoupled QK/PV
        # block sizes; existing call sites that don't pass these stay on 64.
        bn_qk: Int = 0,
        bk_pv: Int = 0,
        # Threshold (in log2 domain) below which the softmax rescale of
        # `O` and `mi` is skipped. Default `-6.0` matches FlashMLA's
        # value.
        skip_correction_threshold: Float32 = -6.0,
        # 0 means "use `padded_q_depth`" (every rotary-carrying call site).
        input_q_depth: Int = 0,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.q_depth = q_depth
        self.rope_depth = q_depth - depth
        self.skip_correction_threshold = skip_correction_threshold

        self.decode_layout_g = decode_layout_g
        # Layout G uses the 1x4 datapath (BM=MMA_M=32); Layout E uses 2x2 (=64).
        if decode_layout_g:
            self.BM = 32
            self.MMA_M = 32
        else:
            self.BM = 64
            self.MMA_M = 64
        self.MMA_PV_N = 256
        # The output writeback path anchors on BN_PV (not BN_QK) so it
        # stays correct when BN_QK is decoupled (e.g. Layout-G-128 BN_QK=128).
        self.BN_PV = self.MMA_PV_N

        self.BN_QK = bn_qk if bn_qk > 0 else 64
        self.BK_PV = bk_pv if bk_pv > 0 else 64
        # QK MMA writes all N cols that softmax tcgen05_ld[repeat=BN_QK/4] reads.
        self.MMA_QK_N = self.BN_QK

        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        self.kv_mma_swizzle_mode = kv_mma_swizzle_mode
        var swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)
        self.padded_q_depth = align_up(q_depth, swizzle_elems)
        self.input_q_depth = (
            input_q_depth if input_q_depth > 0 else self.padded_q_depth
        )

        self.kv_tma_swizzle_mode = (
            TensorMapSwizzle.SWIZZLE_64B if kv_type_size
            == 1 else TensorMapSwizzle.SWIZZLE_128B
        )
        # Split content/rope swizzle modes for per_token_scale_rope_aware:
        # Content is FP8 (1 byte) -> SWIZZLE_64B, Rope is BF16 (2 bytes) -> SWIZZLE_128B.
        # When not per_token_scale_rope_aware, both use the same swizzle as kv_tma_swizzle_mode.
        if per_token_scale_rope_aware:
            self.content_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
            self.rope_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
        else:
            self.content_swizzle_mode = self.kv_tma_swizzle_mode
            self.rope_swizzle_mode = self.kv_tma_swizzle_mode
        self.per_token_scale_rope_aware = per_token_scale_rope_aware
        # Per-token scales for SnapMLA: 1 float32 scale per KV token (sigma_KV).
        # In MLA's absorbed mode K and V both derive from the same latent c_KV,
        # so they share a single per-token quantization scale.
        # Per stage: BN_QK(64) tokens * 1 * sizeof(float32)(4) = 256 bytes.
        if per_token_scale_rope_aware:
            self.per_token_scales_per_stage = self.BN_QK * 1 * 4
        else:
            self.per_token_scales_per_stage = 0
        # All paths use 3 WGs (384 threads) except: the old FP8 converter path
        # (Q=BF16, KV=FP8, both blockwise and tensorwise), which uses 4 WGs
        # (512 threads) for the extra FP8-to-BF16 conversion warpgroup; and
        # the unified native-FP8 kernel, which uses 4 WGs for the re-swizzle
        # staging warpgroup (no dtype conversion, pure FP8->FP8 layout permute).
        var _old_fp8_converter = (
            kv_type_size == 1
            and not native_fp8
            and not per_token_scale_rope_aware
        )
        if _old_fp8_converter or native_fp8_unified_gather:
            self.num_threads = WARPGROUP_SIZE * 4
        else:
            self.num_threads = WARPGROUP_SIZE * 3

        # 4 bytes for the TMEM base pointer
        var smem_use = 4
        self.tmem_used = Self.TMEM_CORR_LI + 1
        self.decoding_warp_split_k = decoding_warp_split_k
        self.page_size = page_size
        self.split_page_size = split_page_size
        self.scale_block_size = scale_block_size
        self.scales_per_token = (
            ceildiv(q_depth, scale_block_size) if scale_block_size > 0 else 0
        )
        # Scale SMEM per stage: e8m0 values (1 byte each) loaded by warp 8.
        # Per stage: BN_QK tokens * scales_per_token * 1 byte.
        # No alignment padding needed: mbarriers placed after scale SMEM
        # require 8-byte alignment, and BN_QK=64 ensures the product is always
        # a multiple of 64 >= 8.
        self.scale_smem_per_stage = (
            self.BN_QK * self.scales_per_token if scale_block_size > 0 else 0
        )
        self.BK_QK = self.padded_q_depth
        self.out_rows = min(self.BM, self.num_q_heads)
        # Q SMEM sizing:
        # - BF16 / old FP8 converter: BF16-sized Q (64x576x2 = 73728 bytes)
        # - Native FP8: FP8-sized Q (64x576x1 = 36864 bytes), Q arrives as FP8
        #   from TMA with SWIZZLE_64B. No conversion needed.
        # - Per-token-scale: split Q = FP8 content (BM*512*1=32768) + BF16 rope (BM*64*2=8192) = 40960 bytes
        if per_token_scale_rope_aware:
            smem_use += (
                self.BM * self.padded_depth * 1 + self.BM * self.rope_depth * 2
            )
        elif native_fp8:
            smem_use += self.BM * self.padded_q_depth * kv_type_size
        else:
            smem_use += self.BM * self.padded_q_depth * dtype_size
        # Three scratch buffers (float32) for softmax running max and li:
        # max uses double-buffering (2 x 128 elements) to avoid a race
        # condition between consecutive loop iterations (write in iteration
        # N+1 can overlap read in iteration N without an extra barrier).
        # li uses a single buffer (only accessed post-loop).
        # Total: 128 threads x 1 element x 4 bytes x 3 buffers = 1536 bytes
        comptime smem_for_max_and_li = 128 * 1 * 4 * 3
        smem_use += smem_for_max_and_li
        # Scale SMEM: for FP8 blockwise, store e8m0 scales (1 byte each).
        # Double-buffered to match the KV pipeline.
        # For tensorwise: no scale SMEM needed.
        # For blockwise: scale_smem_per_stage * 2 stages
        var smem_for_scale: Int
        if scale_block_size > 0:
            # Per stage: BN_QK * scales_per_token * 1 byte (e8m0)
            # Double-buffered: * 2 stages (for blockwise FP8 converter path)
            smem_for_scale = self.scale_smem_per_stage * 2
        else:
            smem_for_scale = 0
        smem_use += smem_for_scale
        # KV SMEM per stage sizing:
        # - BF16 / old FP8 converter: BF16-sized stages (BN_QK * padded_q_depth * dtype_size)
        #   The Softmax writes BF16 P into the KV stage at NumVOBlocks * BlockElems offset.
        # - Native FP8: FP8-sized stages (BN_QK * padded_q_depth * kv_type_size)
        #   P lives in a separate SMEM region (not inside KV stages), so KV stages
        #   can be FP8-sized. This gives 3 stages instead of 2.
        # - Per-token-scale: split KV = FP8 content (BN_QK*512*1=32768) + BF16 rope (BN_QK*64*2=8192) = 40960 bytes/stage
        #   P lives in a separate SMEM region (same as native FP8). P stage = BM*BN_QK*1 = 4096 bytes.
        var smem_per_kv: Int
        var smem_for_p: Int
        # Extra depth given to the MMA-operand KV ring alone; see the
        # unified-gather branch below. 0 on every other path.
        var extra_mma_stages: Int = 0
        if per_token_scale_rope_aware:
            # Per-token-scale: KV stage = content FP8 + rope BF16
            smem_per_kv = (
                self.BN_QK * self.padded_depth * 1
                + self.BN_QK * self.rope_depth * 2
            )
            # P reuses the KV rope SMEM region (no separate P allocation).
            # This is safe because rope is consumed by QK MMA (warp 9)
            # BEFORE softmax produces P, and KV stage barriers prevent
            # the Load warp from overwriting until PV MMA (warp 10) finishes.
            # P (4096 bytes FP8) fits inside rope (8192 bytes BF16).
            # P_i maps to rope stage i: same stage indexing, same barriers.
            # Per-token scales: 256 bytes per stage (64 tokens * 1 * 4 bytes).
            var smem_per_stage_total = (
                smem_per_kv
                + self.per_token_scales_per_stage
                + 6 * Self.mbar_size
            )
            var fixed_barrier_reserve = 11 * Self.mbar_size
            var available = (
                Self.sm100_smem_carveout - smem_use - fixed_barrier_reserve
            )
            var out_bar_count = (self.depth // self.BN_QK) * 2
            var extra_bar_count = ((self.depth // self.BN_QK) - 1) * 2
            available -= (out_bar_count + extra_bar_count) * Self.mbar_size
            self.num_kv_stages = min(
                Self.MAX_TMEM_S_SLOTS, available // smem_per_stage_total
            )
            smem_for_p = 0  # P reuses rope SMEM, no separate allocation
        elif native_fp8:
            smem_per_kv = self.BN_QK * self.padded_q_depth * kv_type_size
            # Native FP8: P lives in a separate SMEM region.
            # Per-stage cost = KV tile + P tile + 6 barriers (kv:2 + s:2 + p:2).
            # Unified-gather adds a second, same-sized linear-gather KV
            # buffer (contiguous SWIZZLE_NONE stage the re-swizzle WG reads
            # from) plus its own 2-barrier producer/consumer pair per stage.
            var p_per_stage = self.BM * self.BK_PV * kv_type_size
            var gather_per_stage = (
                smem_per_kv if native_fp8_unified_gather else 0
            )
            var gather_bar_per_stage = 2 if native_fp8_unified_gather else 0
            var smem_per_stage_total = (
                smem_per_kv
                + p_per_stage
                + gather_per_stage
                + (6 + gather_bar_per_stage) * Self.mbar_size
            )
            # Reserve stage-independent barriers (11) plus output barriers.
            var fixed_barrier_reserve = 11 * Self.mbar_size
            var available = (
                Self.sm100_smem_carveout - smem_use - fixed_barrier_reserve
            )
            var out_bar_count = (self.depth // self.BN_QK) * 2
            var extra_bar_count = ((self.depth // self.BN_QK) - 1) * 2
            available -= (out_bar_count + extra_bar_count) * Self.mbar_size
            self.num_kv_stages = min(
                Self.MAX_TMEM_S_SLOTS, available // smem_per_stage_total
            )
            # Layout G halves Q/P SMEM, so pin to 4 stages.
            if decode_layout_g:
                self.num_kv_stages = 4
            smem_for_p = self.num_kv_stages * (p_per_stage + gather_per_stage)
            # Unified gather stages KV twice: the linear gather destination
            # and the SW64 buffer the MMA reads. Only the second carries the
            # WAR back-edge that closes the steady-state loop, so the loop's
            # rate is bounded by (cycle weight) / (that ring's depth) and the
            # gather ring's depth does not enter it. A symmetric extra stage
            # costs `smem_per_stage_total` and does not fit; one extra SW64
            # stage and its barrier pair does. Spend the leftover there only.
            if native_fp8_unified_gather:
                var extra_kv_stage = smem_per_kv + 2 * Self.mbar_size
                # The sparse dispatch adds idx_bars, idx_smem and the TMEM
                # pointer on top of `smem_used`; reserve them or the deeper
                # ring overflows the carveout at launch rather than here.
                var sparse_reserve = (
                    2 * self.num_kv_stages * Self.mbar_size
                    + 4
                    + self.num_kv_stages * self.BN_QK * 4
                )
                var leftover = (
                    available - self.num_kv_stages * smem_per_stage_total
                )
                if leftover >= extra_kv_stage + sparse_reserve:
                    extra_mma_stages = 1
        else:
            smem_per_kv = self.BN_QK * self.padded_q_depth * dtype_size
            smem_for_p = 0  # P lives inside KV stages for BF16/old FP8
            # now we need to calculate how many slots per K/V we can fit in the remaining memory
            # the carveout reserves 1K for L1 cache so
            # for b200 we have sm100_smem_carveout 233472 - 1024 =  232448 bytes
            self.num_kv_stages = (
                Self.sm100_smem_carveout - smem_use
            ) // smem_per_kv
        self.num_kv_mma_stages = self.num_kv_stages + extra_mma_stages
        smem_use += smem_for_p
        smem_use += self.num_kv_stages * (smem_per_kv)
        # The deeper MMA ring's own tile plus the producer/consumer mbar pair
        # the `fixed_barriers` count below does not cover.
        smem_use += extra_mma_stages * (smem_per_kv + 2 * Self.mbar_size)
        # Per-token scale SMEM: N stages * 256 bytes each (only for per_token_scale_rope_aware)
        smem_use += self.num_kv_stages * self.per_token_scales_per_stage
        # We have the following resources that need smem barriers:

        # bar_write_prod[depth/BN_QK] → 8  producer pipeline - softmax epilogue
        # bar_write_cons[depth/BN_QK] → 8  consumer pipeline - TMA store
        var num_out_barrier = (self.depth // self.BN_QK) * 2
        # total number of barriers is fixed_transaction_barriers + num_out_barrier
        # bar_q → 1           producer pipeline - load consumer - mma
        # bar_kv_ready[2] → 2  consumer pipeline - mma
        # bar_kv_free[2] → 2   producer pipeline - load
        # bar_s_done[2] → 2  producer pipeline - mma
        # bar_s_ready[2] → 2  consumer pipeline - softmax
        # bar_p_done[2] → 2  producer pipeline- softmax
        # bar_p_ready[2] → 2  consumer pipeline - mma
        # bar_correction_done[1] → 1  producer pipeline- softmax
        # bar_correction_ready[1] → 1  consumer pipeline - correction
        # bar_o_done[2] → 2  producer pipeline- MMA PV
        # bar_o_ready[2] → 2  consumer pipeline - Correction
        # corr_done_prod[2] → 2  producer pipeline - correction
        # corr_done_cons[2] → 2  consumer pipeline - softmax

        # Fixed barrier count depends on the path:
        # BF16: 23 barriers (2-stage KV/S/P pipelines)
        # Old FP8 converter: 27 barriers (23 + 4 for convert pipeline)
        # Native FP8 / per_token_scale_rope_aware: 6*N + 11 barriers where N = num_kv_stages
        #   bar_q(1) + kv(2N) + s(2N) + p(2N) + o(4) + c(2) + corr_done(4)
        # Unified-gather adds 2*N more (the linear-gather->re-swizzle pipeline's
        # own producer/consumer barrier pair per stage): 8*N + 11.
        var fixed_barriers: Int
        if per_token_scale_rope_aware or native_fp8:
            fixed_barriers = (
                8 * self.num_kv_stages + 11
            ) if native_fp8_unified_gather else (6 * self.num_kv_stages + 11)
        elif kv_type_size == 1:
            fixed_barriers = (
                27  # Old FP8 converter: 4 extra for convert pipeline
            )
        else:
            fixed_barriers = 23  # BF16: 2-stage pipelines
        smem_use += (fixed_barriers + num_out_barrier) * Self.mbar_size + (
            ((self.depth // self.BN_QK) - 1) * 2 * Self.mbar_size
        )

        # Summary of smem layout:
        # BF16: Q(73728) + KV_stages(2*73728) + max/li(1536) + barriers(23)
        # Old FP8: Q(73728) + KV_stages(2*73728) + max/li(1536) + scale + barriers(27)
        # Native FP8: Q(36864) + KV_stages(N*36864) + P_stages(N*4096) + max/li(1536) + barriers(6N+11)
        #   where N = num_kv_stages (dynamically computed, typically 4)
        # Per-token-scale: Q(40960) + KV_stages(N*40960) + scales(N*256) + max/li(1536) + barriers(6N+11)
        #   Q = FP8 content(32768) + BF16 rope(8192), KV = FP8 content(32768) + BF16 rope(8192)
        #   Per-token scales: N * 256 bytes (64 tokens * 1 scale * 4 bytes float32)
        #   P reuses KV rope SMEM (P_i maps to rope stage i; 4096B FP8 fits in 8192B BF16 rope)
        # max uses double-buffered SMEM (2x128x4=1024B) to avoid race; li uses 1x128x4=512B
        # Plus num_out_barrier = (depth/BN_QK)*2 output barriers,
        # plus ((depth/BN_QK)-1)*2 additional barriers.
        self.smem_used = smem_use

    def supported(self) -> Bool:
        # BM is 32 (Layout G) or 64 (everyone else).
        # `input_q_depth` narrows only the TMA, so it must cover every V block,
        # fit within the SMEM row, and land on a gather column-group boundary.
        var input_row_ok = (
            self.padded_depth <= self.input_q_depth
            and self.input_q_depth <= self.padded_q_depth
            and self.input_q_depth % self.BN_QK == 0
        )
        var base = (
            self.q_depth == 576
            and input_row_ok
            and self.BN_QK == 64
            and (self.BM == 32 or self.BM == 64)
            and self.depth == 512
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )
        if not self.decode_layout_g:
            return base
        # Layout G: BM=MMA_M=32 (1x4 datapath) and >=4 KV stages. The
        # `fold_q` / num_heads * q_len_fold gates live in the dispatcher.
        return (
            base
            and self.BM == 32
            and self.MMA_M == 32
            and self.num_kv_stages >= 4
        )


@always_inline
def rows_owned[config: MLA_SM100_Decode_Config](block_x: Int) -> Int:
    """Returns the number of output rows the CTA at grid x-index block_x
    owns.

    Every head group is out_rows tall except the last, which holds the
    remainder of num_q_heads.

    Parameters:
        config: Decode config supplying the head-group geometry.

    Args:
        block_x: The CTA's grid x-index.

    Returns:
        The number of output rows the CTA owns, at most out_rows.
    """
    return min(config.out_rows, config.num_q_heads - block_x * config.BM)


# ------------------------------------------------------------------------------
# Offset position struct
# ------------------------------------------------------------------------------
struct OffsetPosition[
    config: MLA_SM100_Decode_Config,
    KVLUTType: MHAOperand,
    ragged: Bool,
    is_cache_length_accurate: Bool,
    ValidLengthType: OptionalPointer,
    decoding_warp_split_k: Bool = False,
    sparse: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
](TrivialRegisterPassable):
    """Computes and stores per-CTA row offsets and KV key ranges for the decode kernel.

    Parameters:
        config: Decode config supplying tile sizes and head counts used to
            compute Q and output row offsets.
        KVLUTType: `MHAOperand` providing the KV cache tensor and its
            `cache_length` accessor.
        ragged: When `True`, the valid-lengths tensor is interpreted as
            input row offsets enabling ragged batching.
        is_cache_length_accurate: When `False`, the kernel adds the local
            sequence length to the cache length to compute the total key
            count.
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor.
        decoding_warp_split_k: When `True`, the CTA processes a split-K
            partition of the KV cache (defaults to `False`).
        sparse: When `True`, the kernel iterates over a sparse subset of
            tokens selected by `d_indices` instead of the full KV cache
            (defaults to `False`).
        has_extra_kv: When `True`, sparse attention additionally attends
            to a separate extra-KV cache (defaults to `False`).
        has_variable_topk: When `True`, the sparse top-k length is read
            per batch from `sparse_topk_lengths` instead of using the
            fixed stride (defaults to `False`).
    """

    var seq_len: Int
    var max_seq_len: Int  # q_max_seq_len (padded seq dimension for all batches)
    var num_keys: Int  # Total keys for this batch (full KV cache length)
    var q_row_offset: Int  # Row offset for Q tensor (no split dimension)
    var out_row_offset: Int  # Row offset for output tensor (includes split dimension)
    var split_idx: Int  # Which split partition this CTA handles
    var batch_idx: Int  # Which batch this CTA handles
    var kv_start_row: Int  # Starting KV row for this split
    var num_keys_this_split: Int  # Number of keys this split processes
    var q_token_idx: Int  # Global Q token index for per-token Q scale lookup
    # Logical (pre-sparse-override) total key count for this batch. `num_keys`
    # is overridden below (sparse=True) to the slot count (topk + extra_topk),
    # which only equals the logical count when topk >= actual_tokens; sparse
    # causal masking uses `cache_len_logical()` off this field instead.
    var actual_num_keys: Int

    @always_inline
    def __init__(
        out self,
        k: Self.KVLUTType,
        valid_length: UnsafePointer[
            Scalar[Self.ValidLengthType.dtype], origin=ImmutAnyOrigin
        ],
        max_seq_len: Int,
        num_partitions: Int,
        batch_size: Int,
        # Sparse attention parameters — only used when sparse=True (comptime).
        sparse_indices_stride: Int = 0,
        sparse_topk_lengths: OptionalReg[
            UnsafePointer[Int32, MutAnyOrigin]
        ] = None,
        sparse_extra_indices_stride: Int = 0,
        sparse_extra_topk_lengths: OptionalReg[
            UnsafePointer[Int32, MutAnyOrigin]
        ] = None,
    ):
        self.seq_len = 0
        self.max_seq_len = max_seq_len
        self.num_keys = 0
        self.q_row_offset = 0
        self.out_row_offset = 0
        self.split_idx = 0
        self.batch_idx = 0
        self.kv_start_row = 0
        self.num_keys_this_split = 0
        self.q_token_idx = 0
        self.actual_num_keys = 0

        # Decode block_idx.z into split_idx and batch_idx
        # Grid layout: block_z = batch_size * num_partitions
        # block_idx.z = batch_idx * num_partitions + split_idx
        comptime if Self.decoding_warp_split_k:
            self.batch_idx, self.split_idx = divmod(block_idx.z, num_partitions)
        else:
            self.batch_idx = block_idx.z
            self.split_idx = 0

        comptime if Self.ragged:
            # treat valid_lengths as input_row_offsets
            # Use batch_idx (not block_idx.z) to index into valid_length
            var start_of_seq = Int(valid_length[self.batch_idx])
            var end_of_seq = Int(valid_length[self.batch_idx + 1])
            self.seq_len = end_of_seq - start_of_seq

            # Global Q token index for per-token scale lookup.
            # In ragged mode, Q tokens are packed: token = start_of_seq + seq_idx
            self.q_token_idx = start_of_seq + block_idx.y

            # Q row offset: no split dimension
            # Q shape: (total_tokens * num_heads, depth)
            self.q_row_offset = (
                start_of_seq * Self.config.num_q_heads
                + block_idx.x * Self.config.BM
                + block_idx.y * Self.config.num_q_heads
            )

            # Output row offset: includes split dimension for split-K
            comptime if Self.decoding_warp_split_k:
                # For ragged with split-K, o_accum_split uses PADDED layout:
                # Shape: (num_partitions, batch_size, max_seq_len, num_heads, depth)
                # This must match the combine kernel's read pattern which uses
                # batch_idx * max_seq_len * num_heads as the stride per batch.
                var rows_per_split = (
                    batch_size * max_seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split
                    + self.batch_idx * max_seq_len * Self.config.num_q_heads
                    + block_idx.y * Self.config.num_q_heads
                    + block_idx.x * Self.config.BM
                )
            else:
                self.out_row_offset = self.q_row_offset

        # This is when the sequence length is Fixed
        else:
            self.seq_len = max_seq_len

            # Global Q token index for per-token scale lookup.
            # In fixed mode: token = batch_idx * seq_len + seq_idx
            self.q_token_idx = self.batch_idx * self.seq_len + block_idx.y

            # Q row offset: (batch * seq_len * num_heads, depth)
            # Row = batch_idx * (seq_len * num_heads) + seq_idx * num_heads + head_block * BM
            self.q_row_offset = (
                Self.config.num_q_heads * self.seq_len * self.batch_idx
                + block_idx.x * Self.config.BM
                + block_idx.y * Self.config.num_q_heads
            )

            # Output row offset for split-K:
            # Out shape: (split_k * batch * seq_len * num_heads, depth)
            # Row = split_idx * (batch * seq_len * num_heads) + q_row_offset
            comptime if Self.decoding_warp_split_k:
                var rows_per_split = (
                    batch_size * self.seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split + self.q_row_offset
                )
            else:
                self.out_row_offset = self.q_row_offset

        # Get num_keys from KV cache for this batch
        # Use batch_idx (not block_idx.z) to get the correct cache length
        self.num_keys = k.cache_length(self.batch_idx)

        comptime if not Self.is_cache_length_accurate:
            self.num_keys += self.seq_len

        # Logical total key count, captured before the sparse override
        # below can replace num_keys with the (smaller) sparse slot count.
        self.actual_num_keys = self.num_keys

        # Compute KV range for this split
        # Each split handles a portion of the KV cache: [kv_start_row, kv_start_row + num_keys_this_split)
        comptime if Self.decoding_warp_split_k:
            # Split-page-aligned strategy: only last CTA handles ragged remainder.
            # All other CTAs process complete split_page_size-element chunks.
            comptime page_size = Self.config.split_page_size
            var total_pages = ceildiv(self.num_keys, page_size)
            var pages_per_split = ceildiv(total_pages, num_partitions)

            # Split boundaries are page-aligned
            var start_page = self.split_idx * pages_per_split
            var end_page = min(
                (self.split_idx + 1) * pages_per_split, total_pages
            )

            self.kv_start_row = start_page * page_size
            var kv_end_row = min(end_page * page_size, self.num_keys)
            self.num_keys_this_split = max(kv_end_row - self.kv_start_row, 0)
        else:
            # No split: process all keys starting from row 0
            self.kv_start_row = 0
            self.num_keys_this_split = self.num_keys

        # -------------------------------------------------------------------
        # Sparse attention: override num_keys with topk (clamped to
        # actual_tokens). When sparse=True (comptime) the kernel
        # iterates over a sparse subset of tokens selected by d_indices
        # instead of the full KV cache.
        # -------------------------------------------------------------------
        comptime if Self.sparse:
            # self.num_keys already holds the correct total token count
            # (cache_length + seq_len when _is_cache_length_accurate=False,
            # or just cache_length otherwise). Use it as the upper bound.
            var actual_tokens = self.num_keys

            var topk: Int
            comptime if Self.has_variable_topk:
                topk = Int(
                    sparse_topk_lengths.unsafe_value()[Int(self.batch_idx)]
                )
            else:
                topk = sparse_indices_stride

            # Clamp topk to actual available tokens.
            topk = min(topk, actual_tokens)

            # Extra KV: always-attend tokens from a separate cache.
            var extra_topk: Int = 0
            comptime if Self.has_extra_kv:
                comptime if Self.has_variable_topk:
                    extra_topk = Int(
                        sparse_extra_topk_lengths.unsafe_value()[
                            Int(self.batch_idx)
                        ]
                    )
                else:
                    extra_topk = sparse_extra_indices_stride

            # Override num_keys with the sparse token count.
            self.num_keys = topk + extra_topk

            # Recompute split-K boundaries for the sparse token count.
            var total_topk = topk + extra_topk
            comptime if Self.decoding_warp_split_k:
                comptime page_size = Self.config.split_page_size
                var total_pages_s = ceildiv(total_topk, page_size)
                var pages_per_split_s = ceildiv(total_pages_s, num_partitions)
                var start_page_s = self.split_idx * pages_per_split_s
                var end_page_s = min(
                    (self.split_idx + 1) * pages_per_split_s, total_pages_s
                )
                self.kv_start_row = start_page_s * page_size
                var kv_end_row_s = min(end_page_s * page_size, total_topk)
                self.num_keys_this_split = max(
                    kv_end_row_s - self.kv_start_row, 0
                )
            else:
                self.kv_start_row = 0
                self.num_keys_this_split = total_topk

    @always_inline
    def cache_len(self) -> Int:
        # num_keys is total keys, seq_len is chunk length
        return max(self.num_keys - self.seq_len, 0)

    @always_inline
    def cache_len_logical(self) -> Int:
        # Logical cached-prefix length, unaffected by the sparse num_keys
        # override. Equal to cache_len() in dense mode, and whenever sparse
        # topk >= actual_tokens (topk clamps to the full causal set). Must
        # be used (not cache_len()) for sparse causal masking by LOGICAL
        # key position once topk < actual context length.
        return max(self.actual_num_keys - self.seq_len, 0)

    @always_inline
    def start_pos(self, cache_start_pos: UInt32) -> UInt32:
        # start_pos is the base absolute Q index for this chunk (plus any external base)
        return UInt32(self.cache_len()) + cache_start_pos

    @always_inline
    def q_row_offset_at(self, q_local: Int) -> Int:
        # Per-q_token Q-row offset for TMA load coord (Option A q_len fold,
        # stored q_row_offset bakes in block_idx.y as the q-local term in
        # both ragged and fixed modes, so we swap block_idx.y for q_local.
        return (
            self.q_row_offset
            + (q_local - Int(block_idx.y)) * Self.config.num_q_heads
        )

    @always_inline
    def out_row_offset_at(self, q_local: Int) -> Int:
        # Per-q_token output-row offset for TMA store coord (q_len
        # fold. the stored out_row_offset has
        # exactly one (block_idx.y * num_q_heads) term in every
        # ragged/fixed x split/no-split mode.
        return (
            self.out_row_offset
            + (q_local - Int(block_idx.y)) * Self.config.num_q_heads
        )

    @always_inline
    def q_token_idx_at(self, q_local: Int) -> Int:
        # Global Q-token index for the q_local-th q_token in this CTA's
        # batch. Stored q_token_idx bakes in block_idx.y; swaps
        # block_idx.y for q_local in both ragged and fixed modes.
        return self.q_token_idx + (q_local - Int(block_idx.y))


# ------------------------------------------------------------------------------
# MLA decoding Load fp8 to bf16 ProducerKVPipeline
# ------------------------------------------------------------------------------
struct KVLoad2CvtProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Producer side of the FP8-to-BF16 load-and-convert KV pipeline.

    Parameters:
        dtype: Element type of the FP8 KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    # For blockwise FP8 scaling, warp 8's 32 threads also arrive on the
    # producer mbar after writing scale data to SMEM (release semantics).
    # This eliminates separate named barriers for scale synchronization.
    comptime _load2cvt_num_prod = 1 + (
        32 if Self.config.scale_block_size > 0 else 0
    )
    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages,
        1,
        Self._load2cvt_num_prod,
        WARPGROUP_SIZE + 2,
    ]

    # BF16-stage element count (64*576 = 36864)
    comptime bf16_stage_elems = Self.config.BN_QK * Self.config.q_depth

    # FP8 overlay stride in FP8 elements:
    # lower-half(fp8) + upper-half(fp8) = 2 * bf16_stage_elems
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.KVPipeType

    # IMPORTANT: this pointer must already point to the UPPER HALF (1:fp8) of stage0
    @__allow_legacy_any_origin_fields
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8
        self.pipe.state._phase = 1

    @always_inline
    def init(self):
        self.pipe.init()

    @always_inline
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8 + stage_idx * UInt32(
            Self.fp8_stage_stride_elems
        )

    @always_inline
    def producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @always_inline("nodebug")
    def acquire[*, qk_stage: Int = 0](self):
        self.pipe.producer_acquire[qk_stage]()

    @always_inline("nodebug")
    def commit_step(mut self):
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding Convert fp8 to bf16 ConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVLoad2CvtConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumer side of the FP8-to-BF16 load-and-convert KV pipeline.

    Parameters:
        dtype: Element type of the FP8 KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    # Must match KVLoad2CvtProducer's num_producer for type compatibility.
    comptime _load2cvt_num_prod = 1 + (
        32 if Self.config.scale_block_size > 0 else 0
    )
    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages,
        1,
        Self._load2cvt_num_prod,
        WARPGROUP_SIZE + 2,
    ]

    comptime bf16_stage_elems = Self.config.BN_QK * Self.config.q_depth
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.PipeT

    # points to UPPER HALF (1:fp8) of stage0
    @__allow_legacy_any_origin_fields
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.PipeT,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8

    @always_inline
    def stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8 + idx * UInt32(Self.fp8_stage_stride_elems)

    @always_inline("nodebug")
    def wait(self):
        self.pipe.consumer_wait[0]()

    @always_inline("nodebug")
    def release_all(mut self):
        _ = self.pipe.consumer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding produce bf16 for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVCvt2MmaProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Produces converted BF16 KV tiles for the MMA consumer pipeline.

    Parameters:
        dtype: Element type of the converted KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth

    var pipe: Self.PipeT

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self, pipe: Self.PipeT, smem: SharedMemPointer[Scalar[Self.dtype]]
    ):
        self.pipe = pipe
        self.smem = smem
        self.pipe.state._phase = 1

    @always_inline("nodebug")
    def acquire(self):
        # waits until MMA (2 consumers) released this stage
        self.pipe.producer_acquire[0]()

    @always_inline
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    def stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx = self.pipe.state.index()
        return self.smem + idx * UInt32(Self.kv_stage_elems)

    @always_inline("nodebug")
    def commit_all(mut self):
        # 128 threads arrive on producer mbar
        _ = self.pipe.producer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding consume bf16  for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------
struct KVCvt2MmaConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumes BF16 KV tiles from the convert producer for the MMA pipeline.

    Parameters:
        dtype: Element type of the converted KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @always_inline
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline("nodebug")
    def wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)


# ------------------------------------------------------------------------------
# MLA decoding ProducerKVPipeline
# ------------------------------------------------------------------------------


struct DecodeKVProducer[
    dtype: DType,
    config: MLA_SM100_Decode_Config,
    num_producer: Int = 1,
    num_consumer: Int = 2,
    num_stages: Int = config.num_kv_stages,
](TrivialRegisterPassable):
    """Producer side of the decode KV pipeline that loads KV tiles via TMA.

    Parameters:
        dtype: Element type of the KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
        num_producer: Number of producer threads arriving on each producer
            mbarrier (defaults to 1).
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier (defaults to 2, matching the standard mmaQK+mmaPV
            dual-consumer KV pipeline). Set to e.g. `WARPGROUP_SIZE` when
            the consumer side is a full warpgroup independently arriving
            (not TMA/MMA-elected), as with a manual SMEM-to-SMEM
            re-staging producer/consumer pair.
        num_stages: Ring depth (defaults to `config.num_kv_stages`). Set
            explicitly when a kernel stages KV through two rings of
            different depths.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.num_stages, 1, Self.num_producer, Self.num_consumer
    ]

    # One KV stage = a BN_QK x 576 logical K tile (loaded as
    # NumQKBlocks x BN_QK x BK_QKT).
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth
    comptime kv_stage_bytes = Self.kv_stage_elems * size_of[Self.dtype]()

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @always_inline
    def init(self):
        self.pipe.init()

    @always_inline
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    def producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @always_inline("nodebug")
    def acquire[*, qk_stage: Int = 0](self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire[qk_stage]()

    @always_inline("nodebug")
    def commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.
        self.pipe.state.step()

    @always_inline("nodebug")
    def commit_all(mut self):
        """Explicit-arrive commit for a non-TMA (manual SMEM-write) producer.

        Every thread of the `num_consumer`-wide... actually `num_producer`
        -wide producer role (e.g. a full warpgroup doing a manual SMEM
        re-swizzle store) calls this once; each thread's own local
        `PipelineState` copy steps identically, and the `num_producer`
        independent `arrive()` calls satisfy the mbar's expected count.
        Use this instead of `commit_step()` when there is no TMA hardware
        `expect_bytes` auto-arrival (i.e. the producer wrote SMEM directly,
        not via `cp.async.bulk.tensor`).
        """
        _ = self.pipe.producer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding ConsumerKVPipeline
# ------------------------------------------------------------------------------
struct DecodeKVConsumer[
    dtype: DType,
    config: MLA_SM100_Decode_Config,
    num_producer: Int = 1,
    num_consumer: Int = 2,
    num_stages: Int = config.num_kv_stages,
](TrivialRegisterPassable):
    """Consumer side of the decode KV pipeline that waits for and releases KV stages.

    Parameters:
        dtype: Element type of the KV tiles stored in SMEM.
        config: Decode config supplying `num_kv_stages`, `BN_QK`, and
            `q_depth` for pipeline stage sizing.
        num_producer: Number of producer threads arriving on each producer
            mbarrier (defaults to 1).
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier (defaults to 2, matching the standard mmaQK+mmaPV
            dual-consumer KV pipeline).
        num_stages: Ring depth (defaults to `config.num_kv_stages`). Set
            explicitly when a kernel stages KV through two rings of
            different depths.
    """

    comptime KVPipeType = KVPipelineGeneric[
        Self.num_stages, 1, Self.num_producer, Self.num_consumer
    ]
    # Stage element count tracks the producer (BN_QK x q_depth).
    comptime kv_stage_elems = Self.config.BN_QK * Self.config.q_depth

    var pipe: Self.KVPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        # NOTE: we copy the KVPipeline value – that's how FA4 does it.
        # Both sides keep their own PipelineState; the *barriers* do the real sync.
        self.pipe = pipe
        self.smem = smem

    @always_inline
    def stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    def stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline("nodebug")
    def wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)

    @always_inline("nodebug")
    def release_all(mut self):
        """Explicit-arrive release for a non-MMA (independent-thread) consumer.

        Every thread of the `num_consumer`-wide consumer role (e.g. a full
        warpgroup doing a manual SMEM re-swizzle read) calls this once; the
        `num_consumer` independent `arrive()` calls satisfy the mbar's
        expected count. Use this instead of `release[qk_stage](e)` (which
        uses `elect_mma_arrive` for a single elected thread per warp) when
        every thread independently participates, not just one MMA-eligible
        lane per warp.
        """
        _ = self.pipe.consumer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding KVPipelineGeneric
# ------------------------------------------------------------------------------
struct KVPipelineGeneric[
    num_kv_stages: Int,
    num_qk_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    """
    KVPipeline has `num_kv_stages * num_qk_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_qk_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.

    Parameters:
        num_kv_stages: Number of KV tiles pipelined for the `S = Q@K'`
            and `O += P@V` MMAs.
        num_qk_stages: Number of pipelined sub-MMAs each QK or PV MMA is
            broken into.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_qk_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def init(self):
        # Consumer & Producer mbars: arrived by 1 thread performing TMA/mma
        comptime for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        comptime for i in range(Self.num_stages, Self.num_stages * 2):
            self.mbar[i].init(Int32(Self.num_consumer))

    @always_inline
    def producer_mbar[qk_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + UInt32(Self.num_qk_stages) * idx + qk_stage

    @always_inline
    def consumer_mbar[qk_stage: Int](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset

    @always_inline
    def consumer_mbar[qk_stage: Int](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @always_inline("nodebug")
    def producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """
        Returns the dynamic pipe idx.

        Parameters:
            qk_stage: QK sub-stage index whose consumer mbarrier to wait on
                (defaults to the last QK stage).
        """
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        comptime if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# ------------------------------------------------------------------------------
# MLA decoding MiscMBars for producer and consumer
# ------------------------------------------------------------------------------
struct DecodeSM100MiscMBars[
    num_stages: Int, num_producer: Int, num_consumer: Int
](TrivialRegisterPassable):
    """Manages a generic producer/consumer mbarrier pair for the S, P, C, and O pipelines.

    Parameters:
        num_stages: Number of pipeline slots managed by the barrier pair.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
    """

    @__allow_legacy_any_origin_fields
    var mbar_base: MBarType

    # Generic barrier pair (producer + consumer) with num_stages slots.

    @always_inline
    def __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @always_inline
    def init(self):
        # Layout: [prod[0..num_stages-1], cons[0..num_stages-1]]
        var s_pipe = MBarPipeline[Self.num_stages](self.mbar_base)
        # e.g. for S: 1 producer thread (elect in MMA warpgroup), 128 consumer threads (softmax warpgroup)
        # e.g. for P: 128 producer threads (softmax warpgroup), 1 consumer thread (elect in MMA warpgroup)
        s_pipe.init[
            num_producer=UInt32(Self.num_producer),
            num_consumer=UInt32(Self.num_consumer),
        ]()

    @always_inline
    def producer(self) -> ProducerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base + Self.num_stages}

    @always_inline
    def consumer(self) -> ConsumerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base + Self.num_stages}

    @always_inline
    def end(self) -> MBarType:
        # We consumed 2 * s_num_stages mbars: prod[2] + cons[2]
        return self.mbar_base + 2 * Self.num_stages


# ------------------------------------------------------------------------------
# MLA decoding S pipeline between MMA and Softmax
# ------------------------------------------------------------------------------
########## Producer of the S slot ##########
struct DecodeSProducer(TrivialRegisterPassable):
    """Producer side of the two-stage S pipeline between MMA and softmax."""

    comptime SNumStages = 2
    var pipe: ProducerPipeline[Self.SNumStages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.SNumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @always_inline
    def acquire(self):
        # Wait for softmax to mark this S slot "free"
        self.pipe.acquire()

    @always_inline
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    def commit_mma(mut self, elect: Int32):
        # Signal "S slot is filled" to softmax
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the S slot ##########
struct DecodeSConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage S pipeline between MMA and softmax."""

    comptime SNumStages = 2
    var pipe: ConsumerPipeline[Self.SNumStages]

    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.SNumStages]):
        self.pipe = pipe

    @always_inline
    def wait(self) -> UInt32:
        # Block until MMA has filled the current S slot
        self.pipe.wait()
        return self.pipe.state.index()

    @always_inline
    def release(mut self):
        # Mark this S slot as "consumed" so MMA can reuse it
        self.pipe.release()


# Parameterized versions for N-stage S pipeline (used by native FP8 with 3 stages)
struct DecodeSProducerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized producer side of the S pipeline between MMA and softmax.

    Parameters:
        num_stages: Number of pipeline slots in the S buffer between MMA and
            softmax.
    """

    var pipe: ProducerPipeline[Self.num_stages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.num_stages]):
        self.pipe = pipe

    @always_inline
    def acquire(self):
        self.pipe.acquire()

    @always_inline
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    def commit_mma(mut self, elect: Int32):
        self.pipe.commit_mma(elect)
        self.pipe.step()


struct DecodeSConsumerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized consumer side of the S pipeline between MMA and softmax.

    Parameters:
        num_stages: Number of pipeline slots in the S buffer between MMA and
            softmax.
    """

    var pipe: ConsumerPipeline[Self.num_stages]

    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.num_stages]):
        self.pipe = pipe

    @always_inline
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()

    @always_inline
    def release(mut self):
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding P Pipeline between Softmax and MMA
# ------------------------------------------------------------------------------
########## Producer of the P slot ##########
struct DecodePProducer(TrivialRegisterPassable):
    """Producer side of the two-stage P pipeline between softmax and MMA."""

    comptime PNumStages = 2
    var pipe: ProducerPipeline[Self.PNumStages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Softmax threads collectively wait until MMA has released P
    @always_inline
    def acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase), all 128 threads see the same phase

    # After writing P, all 128 threads call commit()
    @always_inline("nodebug")
    def commit(mut self):
        self.pipe.commit()
        # -> producer_mbar.arrive() (128 arrivals total)
        # -> state.step() (phase toggles for next iteration)

    # optional helper
    @always_inline
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()


########## Consumer of the P slot ##########
struct DecodePConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage P pipeline between softmax and MMA."""

    comptime PNumStages = 2
    var pipe: ConsumerPipeline[Self.PNumStages]

    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Should be called by MMA elect thread only
    @always_inline("nodebug")
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()
        # -> producer_mbar.wait(phase)
        # blocks until 128 Softmax commits complete

    # Also called by MMA elect thread only

    @always_inline("nodebug")
    def release_mma(mut self, elect: Int32):
        # Like KVPipeline.consumer_release but for generic pipeline
        var mbar = self.pipe.consumer_mbar()
        elect_mma_arrive(mbar, elect)
        self.pipe.step()


# Parameterized versions for N-stage P pipeline (used by native FP8 with 3 stages)
struct DecodePProducerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized producer side of the P pipeline between softmax and MMA.

    Parameters:
        num_stages: Number of pipeline slots in the P buffer between softmax
            and MMA.
    """

    var pipe: ProducerPipeline[Self.num_stages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.num_stages]):
        self.pipe = pipe

    @always_inline
    def acquire(self):
        self.pipe.acquire()

    @always_inline("nodebug")
    def commit(mut self):
        self.pipe.commit()

    @always_inline
    def stage_index(self) -> UInt32:
        return self.pipe.state.index()


struct DecodePConsumerN[num_stages: Int](TrivialRegisterPassable):
    """N-stage parameterized consumer side of the P pipeline between softmax and MMA.

    Parameters:
        num_stages: Number of pipeline slots in the P buffer between softmax
            and MMA.
    """

    var pipe: ConsumerPipeline[Self.num_stages]

    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.num_stages]):
        self.pipe = pipe

    @always_inline("nodebug")
    def wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()

    @always_inline("nodebug")
    def release_mma(mut self, elect: Int32):
        var mbar = self.pipe.consumer_mbar()
        elect_mma_arrive(mbar, elect)
        self.pipe.step()


# ------------------------------------------------------------------------------
# MLA decoding O pipeline between MMA and Correction
# ------------------------------------------------------------------------------
########## Producer of the O slot ##########
struct DecodeOProducer(TrivialRegisterPassable):
    """Producer side of the two-stage O pipeline between MMA and correction."""

    comptime ONumStages = 2
    var pipe: ProducerPipeline[Self.ONumStages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.ONumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @always_inline
    def acquire(self):
        # Wait for correction to mark this O slot "free"
        self.pipe.acquire()

    @always_inline
    def slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    def commit_mma(mut self, elect: Int32):
        # Signal "O slot is filled" to correction
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the O slot ##########
struct DecodeOConsumer(TrivialRegisterPassable):
    """Consumer side of the two-stage O pipeline between MMA and correction."""

    comptime ONumStages = 2
    var pipe: ConsumerPipeline[Self.ONumStages]

    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.ONumStages]):
        self.pipe = pipe

    @always_inline
    def wait(self):
        # Block until MMA has filled the current O slot
        self.pipe.wait()
        _ = self.pipe.state.index()

    @always_inline
    def release(mut self):
        # Mark this O slot as "consumed" so MMA can reuse it
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding C Pipeline between Softmax and Correction
# ------------------------------------------------------------------------------
struct DecodeCProducer(TrivialRegisterPassable):
    """Producer side of the single-stage C pipeline between softmax and correction.
    """

    comptime CNumStages = 1
    var pipe: ProducerPipeline[Self.CNumStages]

    @always_inline
    def __init__(out self, pipe: ProducerPipeline[Self.CNumStages]):
        self.pipe = pipe

    # Softmax warpgroup: all 128 threads call acquire() before writing corr scalars
    @always_inline("nodebug")
    def acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase) on correction side (prev iteration)

    # After writing correction scalars for this O:
    @always_inline("nodebug")
    def commit(mut self):
        self.pipe.commit()
        # producer_mbar.arrive() from 128 threads + state.step()


struct DecodeCConsumer(TrivialRegisterPassable):
    """Consumer side of the single-stage C pipeline between softmax and correction.
    """

    comptime CNumStages = 1
    var pipe: ConsumerPipeline[Self.CNumStages]

    # Correction warpgroup: all 128 threads wait until correction scalars are ready
    @always_inline
    def __init__(out self, pipe: ConsumerPipeline[Self.CNumStages]):
        self.pipe = pipe

    @always_inline("nodebug")
    def wait(self):
        # perform producer_mbar.wait(phase)
        self.pipe.wait()

    @always_inline("nodebug")
    def release(mut self):
        # perform consumer_mbar.arrive() from 128 threads + state.step()
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding  pipeline correction is the producer and write is the consumer
# ------------------------------------------------------------------------------


struct OutPipeline[num_out_stages: Int, num_producer: Int, num_consumer: Int](
    TrivialRegisterPassable
):
    """
    OutPipeline has `num_out_stages` stages.
    `num_out_stages` refers to how many output stages we pipeline
    for performing the output store.

    Parameters:
        num_out_stages: Number of output tiles pipelined for the output
            store.
        num_producer: Number of producer threads arriving on each producer
            mbarrier.
        num_consumer: Number of consumer threads arriving on each consumer
            mbarrier.
    """

    comptime num_stages: Int = Self.num_out_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def init(self):
        # Consumer & Producer mbars: arrived by num_producer and num_consumer threads
        comptime for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        comptime for i in range(Self.num_stages):
            (self.mbar + Self.num_stages)[i].init(Int32(Self.num_consumer))

    @always_inline
    def producer_mbar(self) -> MBarType:
        return self.mbar

    @always_inline
    def consumer_mbar(self) -> MBarType:
        return self.mbar + Self.num_stages

    @always_inline("nodebug")
    def producer_acquire(self):
        """
        Returns the dynamic pipe idx.
        """
        var idx = self.state.index()
        self.consumer_mbar()[idx].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_wait(self):
        var idx = self.state.index()
        self.producer_mbar()[idx].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_release[](mut self, e: Int32):
        var idx = self.state.index()
        elect_mma_arrive(self.consumer_mbar() + idx, e)
        self.state.step()

    @always_inline("nodebug")
    def producer_commit(mut self):
        # All 128 producer threads should call this.
        # mbar was initialized with num_producer = WARPGROUP_SIZE,
        # so producer_mbar()[].arrive() must be called by each producer thread.
        var idx = self.state.index()
        _ = self.producer_mbar()[idx].arrive()
        self.state.step()

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_stages)


struct DecodeOutProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Producer side of the output writeback pipeline that stages output tiles in SMEM for TMA store.

    Parameters:
        dtype: Element type of the output tiles stored in SMEM.
        config: Decode config supplying output tile dimensions and stage
            count.
    """

    # Output writeback uses BN_PV/4 (the per-warp stripe width), not BN_QK,
    # so Layout-G-128 (BN_QK=128) still emits 64-col stripes.
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // (Self.config.BN_PV // 4)
    comptime block_per_warp = Self.col_per_warp // (Self.config.BN_PV // 4)
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[Self.num_out_stages, WARPGROUP_SIZE, 1]

    # Per-stage SMEM = BM rows x BN_PV/4 cols.
    comptime out_stage_elems = Self.config.BM * (Self.config.BN_PV // 4)
    comptime out_stage_bytes = Self.out_stage_elems * size_of[Self.dtype]()

    var pipe: Self.OutPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @always_inline
    def init(self):
        # Only producer OR consumer should call init(), not both.
        self.pipe.init()

    @always_inline
    def stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem + stage_offset

    @always_inline
    def producer_mbar(self) -> MBarType:
        return self.pipe.producer_mbar()

    @always_inline("nodebug")
    def acquire(self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire()

    @always_inline("nodebug")
    def commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.

        self.pipe.producer_commit()


struct DecodeOutConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    """Consumer side of the output writeback pipeline that waits for and releases output stages.

    Parameters:
        dtype: Element type of the output tiles stored in SMEM.
        config: Decode config supplying output tile dimensions and stage
            count.
    """

    # Mirrors `DecodeOutProducer` — see there for the BN_PV/4 anchoring.
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // (Self.config.BN_PV // 4)
    comptime block_per_warp = Self.col_per_warp // (Self.config.BN_PV // 4)
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[Self.num_out_stages, WARPGROUP_SIZE, 1]
    comptime out_stage_elems = Self.config.BM * (Self.config.BN_PV // 4)

    var pipe: Self.OutPipeType

    @__allow_legacy_any_origin_fields
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    def __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @always_inline
    def stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem + stage_offset

    @always_inline("nodebug")
    def wait(self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait()

    @always_inline("nodebug")
    def release(mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release(e)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the QK^T MMA with both Q and K operands in SMEM.

    Parameters:
        operand_type: Element type of the Q and K operands in SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BK = Self.config.BK_QK  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QKᵀ
    ]()

    @staticmethod
    @always_inline
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        # Q: 64 x 64, k-major, same swizzle as TMA
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 576 (padded_q_depth)
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 64, k-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 576 columns
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)

    @staticmethod
    @always_inline
    def mma_block[
        *, block_idx: Int, num_blocks: Int
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        # One K-block slice of `mma` (same descriptors, same layouts):
        # absolute k-mmas so the emitted sequence over all blocks matches
        # the bulk form. Block 0 applies c_scale; later blocks accumulate.
        comptime assert Self.num_k_mmas % num_blocks == 0, "uneven K blocks"
        comptime block_k_mmas = Self.num_k_mmas // num_blocks
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=block_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            k_start=block_idx * block_k_mmas,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


struct DecodeSM100PVSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the PV MMA with both P and V operands in SMEM.

    Parameters:
        operand_type: Element type of the P and V operands in SMEM.
        accum_type: Accumulator dtype used for the PV MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_PV_N
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BM = Self.config.BM  # 64
    comptime BN_PV = Self.MMA_N  # 256
    comptime BK = Self.config.BK_PV  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAPVSS = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @always_inline
    def descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is BDepth_max x 64, mn-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.BN_PV,
            BK=Self.BK,  # 64 rows
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=False,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_p_block(
        p_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = p_smem
        # P: 64 x 64, k-major, same swizzle as Q/K
        return smem_descriptor[
            BMN=Self.BM,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,  # P is k-major
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.BM,
            a_BK=Self.BK,
            a_swizzle=Self.config.swizzle_mode,
            a_is_k_major=True,
            b_BMN=Self.BN_PV,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=False,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorTS for QKT (TMEM Q, SMEM K)
# ------------------------------------------------------------------------------
struct DecodeSM100QKTTS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the QK^T MMA with Q in TMEM and K in SMEM.

    Parameters:
        operand_type: Element type of the Q (in TMEM) and K (in SMEM)
            operands.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BK = Self.config.BK_QK  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QKᵀ
    ]()

    @staticmethod
    @always_inline
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 576, k-major, same swizzle as k_tma
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 576 columns
            swizzle_mode=Self.config.kv_mma_swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: UInt32,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws_ts[
            UMMAKind.KIND_F16,
            Self.operand_type,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=Self.config.kv_mma_swizzle_mode,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — native FP8 (KIND_F8F6F4)
# Both Q and K are FP8 e4m3 in SMEM. MMA_K=32, SWIZZLE_64B.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the native FP8 QK^T MMA with both Q and K in FP8 SMEM.

    Parameters:
        operand_type: FP8 element type of the Q and K operands in SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = 32  # FP8 MMA_K
    comptime BK = Self.config.BK_QK  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for FP8 -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QKT
    ]()

    @staticmethod
    @always_inline
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 576 (padded_q_depth)
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 576 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F8F6F4,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.MMA_K,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — split content FP8 (KIND_F8F6F4)
# Content-only: Q_nope and K_nope are FP8 e4m3 in SMEM.
# BK = padded_depth (512), MMA_K=32, SWIZZLE_64B.
# Used by the per-token-scale rope-aware split content/rope kernel.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_Content_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the content-only FP8 QK^T MMA used by the per-token-scale rope-aware kernel.

    Parameters:
        operand_type: FP8 element type of the content Q and K operands in
            SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = 32  # FP8 MMA_K
    comptime BK = Self.config.padded_depth  # 512 (content only)
    comptime num_k_mmas = Self.BK // Self.MMA_K  # 16
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for FP8 -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QK^T
    ]()

    @staticmethod
    @always_inline
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 512 (padded_depth)
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 512 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F8F6F4,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.MMA_K,
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT — split rope BF16 (KIND_F16)
# Rope-only: Q_rope and K_rope are BF16 in SMEM.
# BK = rope_depth (64), MMA_K=16, SWIZZLE_128B.
# Used by the per-token-scale rope-aware split content/rope kernel.
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS_Rope_BF16[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    """Tensor accumulator for the rope-only BF16 QK^T MMA used by the per-token-scale rope-aware kernel.

    Parameters:
        operand_type: BF16 element type of the rope Q and K operands in
            SMEM.
        accum_type: Accumulator dtype used for the QK^T MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = 16  # BF16 MMA_K
    comptime BK = Self.config.rope_depth  # 64 (rope only)
    comptime num_k_mmas = Self.BK // Self.MMA_K  # 4
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for BF16 -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QK^T
    ]()

    @staticmethod
    @always_inline
    def descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = q_smem
        return smem_descriptor[
            BMN=Self.config.BM,  # 64 rows
            BK=Self.BK,  # 64 (rope_depth)
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.config.BN_QK,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F16,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.config.BM,
            a_BK=Self.BK,
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_is_k_major=True,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for PV — native FP8 (KIND_F8F6F4)
# Both P and V are FP8 e4m3 in SMEM. MMA_K=32, SWIZZLE_64B.
# ------------------------------------------------------------------------------
struct DecodeSM100PVSS_FP8[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
    p_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
](TrivialRegisterPassable):
    """Tensor accumulator for the native FP8 PV MMA with both P and V in FP8 SMEM.

    Parameters:
        operand_type: FP8 element type of the P and V operands in SMEM.
        accum_type: Accumulator dtype used for the PV MMA result in TMEM.
        config: Decode config supplying MMA tile dimensions and swizzle
            modes.
        p_swizzle: SMEM swizzle mode applied to the P operand descriptor
            (defaults to `SWIZZLE_64B`).
    """

    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_PV_N
    comptime MMA_K = 32  # FP8 MMA_K
    comptime BM = Self.config.BM  # 64
    comptime BN_PV = Self.MMA_N  # 256
    comptime BK = Self.config.BK_PV  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- Instruction descriptor for FP8 -----
    comptime UMMAPVSS = UMMAInsDescriptor[UMMAKind.KIND_F8F6F4].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @always_inline
    def descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        return smem_descriptor[
            BMN=Self.BN_PV,
            BK=Self.BK,  # 64 rows
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            is_k_major=False,
        ](base)

    @staticmethod
    @always_inline
    def descriptor_p_block(
        p_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = p_smem
        return smem_descriptor[
            BMN=Self.BM,  # 64 rows
            BK=Self.BK,  # 64 columns
            swizzle_mode=Self.p_swizzle,
            is_k_major=True,  # P is k-major
        ](base)

    @staticmethod
    @always_inline
    def mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            UMMAKind.KIND_F8F6F4,
            Self.operand_type,
            Self.operand_type,
            a_BMN=Self.BM,
            a_BK=Self.BK,
            a_swizzle=Self.p_swizzle,
            a_is_k_major=True,
            b_BMN=Self.BN_PV,
            b_BK=Self.BK,
            b_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            b_is_k_major=False,
            num_k_mmas=Self.num_k_mmas,
            operand_size=Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.MMA_K,
        ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# Helper functions for writing from local memory to shared memory using swizzle
# ------------------------------------------------------------------------------


@always_inline
def write_bf16x2_row_to_smem_chunked[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    chunk_size: Int = 16,
    scale_needed: Bool = False,
](
    shared_mem: UnsafePointer[
        Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED
    ],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size]()],
    col_start: Int,
    row_start: Int,
    scale: Scalar[in_dtype] = 1.0,
):
    """Chunked write with optional scaling. Reduces register pressure.

    Parameters:
        local_tile_size: Number of source elements held by each lane in
            the local tile; must be divisible by `chunk_size`.
        out_dtype: Output dtype written to SMEM, packed as `bf16x2` stores.
        in_dtype: Input dtype of the per-lane local register tile.
        config: Decode config supplying `BN_PV` used to size the SMEM
            row stripe and the ldmatrix swizzle.
        chunk_size: Number of source elements written per chunk (must be
            a multiple of 8; defaults to 16).
        scale_needed: When `True`, multiply each register by `scale`
            before casting to `out_dtype` (defaults to `False`).

    Args:
        shared_mem: Pointer to the destination shared memory region in
            swizzled `bf16x2` layout.
        local_mem: Per-lane local tensor holding the source tile in
            `row_major[local_tile_size]` layout.
        col_start: Starting column offset within the SMEM row stripe.
        row_start: Starting row offset within the SMEM row stripe.
        scale: Scalar multiplier applied before cast when `scale_needed`
            is `True` (defaults to `1.0`).
    """
    comptime num_chunks = local_tile_size // chunk_size
    comptime groups_per_chunk = chunk_size // 8
    comptime total_groups = num_chunks * groups_per_chunk

    # SMEM row width = BN_PV/4 (per-warp stripe).
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size=config.BN_PV // 4,
        log2_vector_width=3,
    ]()

    # Precompute all swizzle offsets before the loop
    var phys_offsets = StaticTuple[Int, total_groups]()

    comptime for i in range(total_groups):
        comptime chunk_idx, group_idx = divmod(i, groups_per_chunk)
        comptime col_offset = chunk_idx * chunk_size + group_idx * 8
        var logical_elem = (
            row_start * (config.BN_PV // 4) + col_start + col_offset
        )
        phys_offsets[i] = swz(logical_elem)

    var lmv = local_mem.vectorize[8]()

    comptime for chunk in range(0, num_chunks):
        comptime for g in range(0, groups_per_chunk):
            # Compute the correct index into the vectorized view
            # vec_idx accounts for both chunk offset and position within chunk
            comptime vec_idx = chunk * groups_per_chunk + g

            var vec_val = lmv[vec_idx]

            comptime if scale_needed:
                vec_val *= scale

            var bf16_vec = vec_val.cast[out_dtype]()
            var packed = bitcast[.uint32, 4](bf16_vec)
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=out_dtype](
                shared_mem,
                phys_offsets[vec_idx],
                packed,
            )


@always_inline
def write_fp8_row_to_smem_chunked[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    chunk_size: Int = 16,
    scale_needed: Bool = False,
    row_size: Int = config.BN_QK,
    swizzle_kind: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_64B,
](
    shared_mem: UnsafePointer[
        Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED
    ],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size]()],
    col_start: Int,
    row_start: Int,
    scale: Scalar[in_dtype] = 1.0,
):
    """Writes float32 data to SMEM as FP8 with FP8-byte swizzle.

    Each group writes 16 FP8 elements = 16 bytes = 4 x uint32.

    Parameters:
        local_tile_size: Number of fp32 source elements held by this lane.
        out_dtype: Output FP8 dtype written to SMEM.
        in_dtype: Input fp32 register dtype.
        config: MLA decode config (used for default `row_size`).
        chunk_size: Number of fp32 elements per chunk (must be a multiple of 16).
        scale_needed: When True, multiply each register by `scale` before cast.
        row_size: Logical SMEM row width in FP8 elements. Defaults to
            `config.BN_QK` for backward compatibility with Layout-G-64 / Layout-E.
            Layout-G-128 passes `row_size = BK_PV = 128`.
        swizzle_kind: Swizzle scheme used by the consuming MMA descriptor.
            Defaults to `SWIZZLE_64B` (existing behaviour). Layout-G-128 must
            pass `SWIZZLE_128B` so the writer's address pattern matches the
            new K=128 P-tile descriptor.

    Args:
        shared_mem: Pointer to the destination shared memory region in
            FP8-byte-swizzled layout.
        local_mem: Per-lane local tensor holding the float32 source tile
            in `row_major[local_tile_size]` layout.
        col_start: Starting column offset within the SMEM row.
        row_start: Starting row offset within the SMEM row.
        scale: Scalar multiplier applied to each register before cast
            when `scale_needed` is `True` (defaults to `1.0`).
    """
    comptime num_chunks = local_tile_size // chunk_size
    comptime groups_per_chunk = chunk_size // 16  # 16 FP8 elements per store
    comptime total_groups = num_chunks * groups_per_chunk

    # `make_ldmatrix_swizzle` produces the right SWIZZLE_64B / SWIZZLE_128B
    # address pattern when given the matching row_size and
    # log2_vector_width=4 (16 FP8 elements = 16 B per store).
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size=row_size,
        log2_vector_width=4,  # log2(16) for 16 FP8 elements
    ]()

    # Precompute all swizzle offsets before the loop
    var phys_offsets = StaticTuple[Int, total_groups]()

    comptime for i in range(total_groups):
        comptime chunk_idx, group_idx = divmod(i, groups_per_chunk)
        comptime col_offset = chunk_idx * chunk_size + group_idx * 16
        var logical_elem = row_start * row_size + col_start + col_offset
        phys_offsets[i] = swz(logical_elem)

    var lmv = local_mem.vectorize[4]()

    comptime for chunk in range(0, num_chunks):
        comptime for g in range(0, groups_per_chunk):
            comptime vec_base = (chunk * groups_per_chunk + g) * 4
            # Process 16 FP8 elements: load 4x float32, cast to 4x FP8
            # But we need to pack 16 FP8 values into 4 uint32 registers.
            # Load 4 groups of 4 float32, cast each group to 4 FP8, pack into uint32.
            var packed = SIMD[.uint32, 4](0)
            comptime for sub in range(4):
                var vec_val = lmv[vec_base + sub]
                comptime if scale_needed:
                    vec_val *= scale
                var fp8_vec = vec_val.cast[out_dtype]()
                packed[sub] = bitcast[.uint32, 1](fp8_vec)

            st_shared_v4_b32_at_fp8_elem_off[out_dtype=out_dtype](
                shared_mem,
                phys_offsets[chunk * groups_per_chunk + g],
                packed,
            )


@always_inline
def st_shared_v4_b32_at_fp8_elem_off[
    out_dtype: DType
](
    dst_fp8: UnsafePointer[
        Scalar[out_dtype], MutAnyOrigin, address_space=.SHARED
    ],
    elem_off: Int,  # FP8 element offset
    packed: SIMD[.uint32, 4],
):
    """Stores four uint32 values to shared memory at an FP8 element offset.

    Parameters:
        out_dtype: FP8 element type of the shared memory destination.

    Args:
        dst_fp8: Shared memory pointer typed to `out_dtype` elements.
        elem_off: Offset from `dst_fp8` in `out_dtype` elements to the
            first stored value.
        packed: Four uint32 registers holding the packed FP8 values to
            store.
    """
    # Delegates to the shared `st_shared_v4_b32` (moved to attention_utils);
    # `elem_off` is in fp8 elements (the pointer is fp8-typed).
    st_shared_v4_b32(dst_fp8, elem_off, packed)


@always_inline
def ld_shared_v4_u32(
    src_u8: UnsafePointer[mut=True, UInt8, _, address_space=.SHARED],
    byte_off: Int,
) -> SIMD[.uint32, 4]:
    """Loads four contiguous uint32 values from shared memory at a byte offset.

    Args:
        src_u8: Byte-typed shared memory pointer to the base of the load.
        byte_off: Byte offset from `src_u8` to the first uint32 to load.
    """
    var addr = src_u8 + byte_off
    var result = inlined_assembly[
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        # 4 outputs (return struct has 4 elems) + 1 input (addr)
        constraints="=r,=r,=r,=r,l",
        has_side_effect=False,
    ](addr)
    return SIMD[.uint32, 4](result[0], result[1], result[2], result[3])


@always_inline
def cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
    *,
    fp8_dtype: DType,
    out_dtype: DType,
](w0: UInt32, w1: UInt32,) -> SIMD[.uint32, 4]:
    """Converts eight FP8 values packed in two uint32 registers to eight BF16 values packed in four uint32 registers.

    Parameters:
        fp8_dtype: FP8 element type of the source values packed in `w0`
            and `w1`.
        out_dtype: Target element type of the converted output values
            (`bfloat16`).

    Args:
        w0: Lower uint32 register holding the first four packed FP8
            values.
        w1: Upper uint32 register holding the next four packed FP8
            values.
    """
    var u32x2: SIMD[.uint32, 2] = SIMD[.uint32, 2](w0, w1)
    var fp8x8: SIMD[fp8_dtype, 8] = bitcast[fp8_dtype, 8](u32x2)
    var bf16x8: SIMD[out_dtype, 8] = fp8x8.cast[out_dtype]()
    return bitcast[.uint32, 4](bf16x8)


@always_inline
def cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4[
    *,
    fp8_dtype: DType,
    out_dtype: DType,
](w: SIMD[.uint32, 4]) -> StaticTuple[SIMD[.uint32, 4], 2]:
    """Converts 16 FP8 bytes (one v4.b32 load) to 16 packed BF16 values."""
    return StaticTuple[SIMD[.uint32, 4], 2](
        cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
            fp8_dtype=fp8_dtype, out_dtype=out_dtype
        ](w[0], w[1]),
        cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
            fp8_dtype=fp8_dtype, out_dtype=out_dtype
        ](w[2], w[3]),
    )


@always_inline
def st_shared_v4_b32_at_bf16_elem_off[
    out_dtype: DType
](
    dst_bf16: UnsafePointer[
        mut=True, Scalar[out_dtype], _, address_space=.SHARED
    ],
    elem_off: Int,  # bf16 element offset
    packed: SIMD[.uint32, 4],
):
    """Stores four uint32 values to shared memory at a BF16 element offset.

    Parameters:
        out_dtype: Element type of the shared-memory destination (`bfloat16`).

    Args:
        dst_bf16: Shared memory pointer typed to `out_dtype` elements.
        elem_off: Offset from `dst_bf16` in `out_dtype` elements to the
            first stored value.
        packed: Four uint32 registers holding the packed values to store.
    """
    # Delegates to the shared `st_shared_v4_b32` (moved to attention_utils);
    # `elem_off` is in bf16 elements (the pointer is bf16-typed).
    st_shared_v4_b32(dst_bf16, elem_off, packed)


@always_inline
def e8m0_to_bf16_broadcast(scale_byte: UInt8) -> UInt32:
    """Convert an e8m0 scale byte to a bf16 value broadcast into both halves of a uint32.

    e8m0 format: value = 2^(byte - 127). The bf16 representation is
    obtained by left-shifting the 8-bit exponent by 7 to place it in the
    bf16 exponent field (bits 7-14), with sign=0 and mantissa=0.
    Broadcasting into both halves of a uint32 prepares the value for
    use with the packed bf16x2 multiply instruction.

    Args:
        scale_byte: E8M0 exponent byte encoding the scale as
            `2^(byte - 127)`.
    """
    var bf16_bits = UInt16(scale_byte) << 7
    return UInt32(bf16_bits) | (UInt32(bf16_bits) << 16)


@always_inline
def hmul2_bf16x8_by_scalar[
    out_dtype: DType,
](packed: SIMD[.uint32, 4], scale_bf16: UInt32) -> SIMD[.uint32, 4]:
    """Multiply 8 packed bf16 values (in 4 uint32 registers) by a bf16x2 scalar broadcast.

    Parameters:
        out_dtype: Element type of the packed bf16 values (`bfloat16`).

    Args:
        packed: Four uint32 registers holding the eight packed bf16 values
            to scale.
        scale_bf16: Bf16x2 scalar broadcast in a uint32, as produced by
            `e8m0_to_bf16_broadcast`.
    """
    var res = type_of(packed)()

    comptime for i in range(packed.length):
        res[i] = inlined_assembly[
            "mul.rn.bf16x2 $0, $1, $2;",
            UInt32,
            constraints="=r,r,r",
            has_side_effect=False,
        ](packed[i], scale_bf16)
    return res


# --------------------------------------------------------------------------
# MLA decoding softmax Pipeline
# --------------------------------------------------------------------------
@always_inline
def clamped_index_coordinate(
    var prompt_idx: UInt32,
    var q_head_idx: UInt32,
    var q_idx_abs: UInt32,
    var col: UInt32,
    var tile_key_base: UInt32,
    var num_keys: Int,
    var cache_start_pos: UInt32,
) -> IndexList[4, element_type=.uint32]:
    """Builds a four-component index coordinate with the key index clamped to the last valid key.

    Args:
        prompt_idx: Batch index of the prompt this coordinate belongs to.
        q_head_idx: Query head index of this coordinate.
        q_idx_abs: Absolute query token index used by the mask.
        col: Column offset within the current KV tile.
        tile_key_base: Starting key index of the current KV tile in
            global KV coordinates, before `cache_start_pos` is added.
        num_keys: Total number of valid keys in the KV cache for this
            batch.
        cache_start_pos: External base offset added to key indices to
            compute absolute positions.
    """
    # Global key index (column) for this element
    var score_col: UInt32 = tile_key_base + col
    var k_idx_abs: UInt32 = score_col + cache_start_pos
    # Clamp k to last valid key so MaterializedMask never reads OOB.
    var last_k_abs: UInt32 = cache_start_pos + UInt32(max(num_keys - 1, 0))
    var k_idx_abs_safe: UInt32 = min(k_idx_abs, last_k_abs)
    return IndexList[4, element_type=.uint32](
        Int(prompt_idx),
        Int(q_head_idx),
        Int(q_idx_abs),
        Int(k_idx_abs_safe),
    )


struct MLA_SM100_Decode_Common[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_dtype: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    """Provides the shared softmax, correction, and store logic for SM100 MLA decode kernels.

    Parameters:
        q_type: Element type of the Q and P operands stored in SMEM.
        KVLUTType: `MHAOperand` providing the KV cache tensor and its
            element type.
        output_dtype: Element type of the output tile written back via TMA
            (must be `bfloat16`).
        SplitAccumType: `OptionalPointer` type wrapping the split-K LSE
            accumulator buffer.
        MaskType: `MHAMask` type applied to the attention scores.
        config: Decode config supplying tile sizes, swizzle modes, and
            TMEM/SMEM layout.
        ValidLengthType: `OptionalPointer` type wrapping the per-batch
            valid-sequence-length tensor.
        _is_cache_length_accurate: When `False`, the kernel adds the local
            sequence length to the cache length to compute the total key
            count (defaults to `False`).
        ragged: When `True`, the valid-lengths tensor is interpreted as
            input row offsets enabling ragged batching (defaults to
            `False`).
    """

    comptime kv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    # Output tile width uses BN_PV/4 (the per-warp stripe), not BN_QK.
    comptime output_tile_width = ((Self.config.BN_PV // 4) // 2) * (
        4 // size_of[Self.output_dtype]()
    )
    # O: 128 x 256
    comptime O_M = Self.config.BM * 2  # 128
    comptime O_N = Self.config.padded_depth // 2  # 256

    # S: 128 x 32
    comptime S_M = Self.config.BM * 2  # 128
    comptime S_N = Self.config.BN_QK // 2  # 32
    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    # --------------------------------------------------------------------------
    # PDL early exit cleanup for split-K CTAs with no work.
    # Writes -inf to LSE so the combine kernel gives this split zero weight,
    # then calls barrier() + launch_dependent_grids().
    #
    # o_accum_split is deliberately left untouched: the combine kernel's
    # `select` guard (scale != 0) keeps uninitialised memory out of the
    # result when scale == 0 (i.e. LSE == -inf).
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def pdl_early_exit[
        # When True, the fold caller owns all q_max_seq_len LSE
        # slots and must emit -inf to each.
        fold_q: Bool = False,
    ](
        split_idx: Int,
        batch_idx: Int,
        max_seq_len: Int,
        batch_size: Int,
        lse_accum_split_ptr: Self.SplitAccumType,
        # Explicit seq_idx for fold callers iterating q_local 0..q_len_fold-1.
        # Under fold grid.y=1 so block_idx.y can't address all seq slots.
        # Only consumed when fold_q=True.
        seq_idx_fold: UInt32 = 0,
    ):
        var tid = thread_idx.x

        # -- 1. Write -inf to LSE so combine gives this split zero weight --
        # LSE layout: (num_splits, batch_size, max_seq_len, num_heads)
        # Use max_seq_len (not per-batch seq_len) for strides to match
        # the PADDED buffer layout and the combine kernel's read pattern.
        var head_start = block_idx.x * Self.config.BM
        var seq_idx: Int
        comptime if fold_q:
            seq_idx = Int(seq_idx_fold)
        else:
            seq_idx = block_idx.y
        var stride_seq = Self.config.num_q_heads
        var stride_batch = max_seq_len * stride_seq
        var stride_split = batch_size * stride_batch
        var neg_inf_val = min_or_neg_inf[Self.AccumType]()

        # First BM threads each write one head's LSE
        if tid < Self.config.BM:
            var head_idx = head_start + tid
            if head_idx < Self.config.num_q_heads:
                var lse_offset = (
                    split_idx * stride_split
                    + batch_idx * stride_batch
                    + seq_idx * stride_seq
                    + head_idx
                )
                var lse_ptr = rebind[
                    UnsafePointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                ](lse_accum_split_ptr.value())
                lse_ptr[lse_offset] = neg_inf_val

        # -- 2. Barrier + PDL signal, then return --
        barrier()
        launch_dependent_grids()

    # --------------------------------------------------------------------------
    # MLA decoding load_q and load_kv function
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def load_kv(
        tma: KVTMATile[
            dtype=Self.kv_type,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
            BN=Self.config.BN_QK,  # tile_m: 64 (Layout-E / Layout-G-64) or 128 (Layout-G-128)
            BK=Self.config.input_q_depth,  # the row as stored
        ],
        smem: SharedMemPointer[Scalar[Self.kv_type]],
        mbar: MBarType,
        col_start: Int,
        row_start: Int,
    ):
        # TMA only uses .ptr from the destination — layout is irrelevant
        # (swizzle is in the TMA descriptor). Use flat row_major TileTensor.
        comptime kv_elements = Self.config.BN_QK * Self.config.input_q_depth
        comptime kv_tt_layout = tt_row_major[kv_elements]()
        var smem_tensor = TileTensor[
            Self.kv_type, type_of(kv_tt_layout), address_space=.SHARED
        ](smem, kv_tt_layout)
        tma.async_copy_3d(smem_tensor, mbar[], (col_start, 0, row_start))

    @staticmethod
    @always_inline
    def load_q(
        tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # tile_m =64
            BK=Self.config.input_q_depth,  # the row as stored
            swizzle_mode=Self.config.swizzle_mode,
        ],
        smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar: MBarType,
        col_start: Int,
        row_start: Int,
    ):
        comptime q_elements = Self.config.BM * Self.config.input_q_depth
        comptime q_tt_layout = tt_row_major[q_elements]()
        var smem_tensor = TileTensor[
            Self.q_type, type_of(q_tt_layout), address_space=.SHARED
        ](smem, q_tt_layout)

        tma.async_copy(smem_tensor, mbar[], (col_start, row_start))

    @staticmethod
    @always_inline
    def apply_mask[
        half_load: Int,
        NonCausalMask: Bool,
        CausalMask: Bool,
        # when > 0, causal_limit is per-row
        # and derived from (score_row // fold_q_num_heads) + cache_len + 1,
        # treating score_row as the fold tile's per-thread row index.
        fold_q_num_heads: Int = 0,
        # When > 0, additionally mask out keys that are MORE than
        # SlidingWindowSize positions before the current query (i.e. clear
        # the low bits of mask_bits below `causal_limit - SlidingWindowSize`).
        # Implies causal upper bound (CausalMask must be True).
        SlidingWindowSize: Int = 0,
        # When True, `col_base + i` (the tile-relative gather SLOT) is not a
        # logical key position -- it indexes a score-sorted sparse top-k
        # selection. Causality must instead be decided by looking up each
        # slot's LOGICAL key position via `logical_indices` and comparing it
        # to `causal_limit`, rather than by counting slots. Requires
        # CausalMask=True and SlidingWindowSize=0 (checked by the caller).
        # See Kernels/claude_kb -- sparse-mla-decode-causal-slot-bug.
        SparseCausalLogical: Bool = False,
    ](
        tiles_done: Int,
        col0: Int,
        num_keys: Int,
        s_row: LocalTensor[Self.AccumType, row_major[half_load]()],
        mask: Self.MaskType,
        prompt_idx: UInt32,
        q_head_idx: UInt32,
        score_row: UInt32,
        cache_len: Int,
        start_pos: UInt32,
        cache_start_pos: UInt32,
        kv_start_row: Int = 0,  # Starting KV row for split-K (0 for non-split)
        # Raw (pre-physical-remap) logical sparse indices, same
        # [total_q_tokens, indices_stride] layout as the physical `d_indices`
        # gather buffer, with -1 for invalid/padding slots. Only read when
        # SparseCausalLogical=True.
        logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
        logical_indices_stride: Int = 0,
        q_token_idx: Int = 0,
        # Valid per-q_token length of `logical_indices` (the main sparse topk,
        # excluding extra_d_indices). The physical gather clamps overflow slots
        # to a duplicate in-range row, but `logical_indices` has no such row,
        # so slots >= this bound must be masked without a read or they run off
        # the end of the q_token's row.
        logical_indices_len: Int = 0,
    ) -> Scalar[Self.AccumType]:
        # Tile / column base this thread covers in num_keys in global KV cache
        # For split-K: kv_start_row + tiles_done * BN_QK gives global position
        # (was BN_QK; matches actual KV tile stride at Layout-G-128)
        # For non-split: kv_start_row=0, so this is just tiles_done * BN_QK
        var tile_key_base: Int = kv_start_row + tiles_done * Self.config.BN_QK
        # first key index for this thread
        var col_base: Int = tile_key_base + col0

        # Per-row causal masking for chunked decode
        # Allowed keys for this query row in a chunked causal decode:
        # cache_len + score_row + 1
        var causal_limit: Int

        comptime if CausalMask:
            comptime if fold_q_num_heads > 0:
                # Fold: score_row is the fold tile row index; q_local =
                # score_row // num_q_heads selects the causal horizon.
                causal_limit = (
                    cache_len + (Int(score_row) // fold_q_num_heads) + 1
                )
            else:
                causal_limit = cache_len + Int(score_row) + 1
        else:
            causal_limit = num_keys
        var keys_remaining = causal_limit - col_base
        var n_valid = max(min(keys_remaining, half_load), 0)
        # Build mask_bits with lowest n_valid bits = 1
        var mask_bits_64: UInt64 = (UInt64(1) << UInt64(n_valid)) - UInt64(1)
        var mask_bits: UInt32 = UInt32(mask_bits_64 & UInt64(0xFFFF_FFFF))

        # Sliding window: also clear bits BELOW the per-row lower limit.
        # Per-row lower limit (in global KV index) = causal_limit -
        # SlidingWindowSize. Bits in mask_bits correspond to columns
        # [col_base, col_base + half_load), so bit i maps to global key
        # index `col_base + i`. Clear bits where `col_base + i <
        # per_row_lo`, i.e. `i < per_row_lo - col_base`. Clamp to
        # [0, half_load].
        comptime if SlidingWindowSize > 0:
            var per_row_lo: Int = causal_limit - SlidingWindowSize
            var n_invalid_low: Int = max(per_row_lo - col_base, 0)
            n_invalid_low = min(n_invalid_low, half_load)
            var low_mask_64: UInt64 = (
                UInt64(1) << UInt64(n_invalid_low)
            ) - UInt64(1)
            var low_mask: UInt32 = UInt32(low_mask_64 & UInt64(0xFFFF_FFFF))
            mask_bits &= ~low_mask

        # Initialize the per-row running max to the finite mask sentinel so
        # an all-masked tile produces a finite `current_max` (= MASK_VALUE)
        # instead of true -inf — keeps later softmax math NaN-free.
        var current_max: Scalar[Self.AccumType] = MASK_VALUE

        comptime assert not (
            SparseCausalLogical and SlidingWindowSize > 0
        ), "SparseCausalLogical does not support sliding window."

        # Runtime null guard: callers that request the logical-position path
        # (comptime) but don't actually thread a logical index buffer (e.g.
        # direct-kernel tests, or sparse kernels not yet wired for it) fall
        # back to the prior slot-count behavior instead of a null deref.
        var use_sparse_causal_logical: Bool = False
        comptime if SparseCausalLogical:
            use_sparse_causal_logical = Bool(logical_indices)
            if use_sparse_causal_logical:
                # Decide visibility for the whole group up front and hand the
                # element loop the same `mask_bits` word every other mask
                # produces. Deciding it per element instead put the read
                # INSIDE the loop's branch, so each of the half_load reads
                # paid its own memory latency. The slots are contiguous in
                # `i`; reading them in batches lets a batch share one
                # latency, and a short batch keeps the live set from growing
                # by all half_load positions.
                #
                # Slots at or past `logical_indices_len` are the gather's
                # clamped overflow and must be masked WITHOUT a read (see
                # that argument's docstring). Carrying that bound on the LOOP
                # rather than per element keeps the set of addresses read
                # here exactly the set the per-element form read.
                comptime logical_batch = 8
                var n_readable = max(
                    min(logical_indices_len - col_base, half_load), 0
                )
                var logical_row = logical_indices.unsafe_value() + (
                    q_token_idx * logical_indices_stride + col_base
                )
                # `pos <= causal_limit - 1` in 32 bits is `pos <
                # causal_limit` in Int, for every position `logical_indices`
                # can represent: causal_limit >= 1 here (CausalMask is
                # required), and saturating the bound keeps a horizon wider
                # than Int32 admitting every slot, as the Int compare did.
                var causal_last = SIMD[.int32, logical_batch](
                    Int32(min(causal_limit - 1, Int(Int32.MAX)))
                )
                # -1 marks a padding slot, so the low bound rejects it.
                comptime lowest_pos = SIMD[.int32, logical_batch](0)
                var logical_bits: UInt32 = 0
                var slot = 0
                while slot + logical_batch <= n_readable:
                    var pos = logical_row.load[
                        width=logical_batch, alignment=4
                    ](slot)
                    var visible = pos.ge(lowest_pos) & pos.le(causal_last)
                    var batch_bits: UInt32 = 0
                    comptime for j in range(logical_batch):
                        if visible[j]:
                            batch_bits |= UInt32(1) << UInt32(j)
                    logical_bits |= batch_bits << UInt32(slot)
                    slot += logical_batch
                while slot < n_readable:
                    var pos = logical_row[slot]
                    if pos >= 0 and pos <= causal_last[0]:
                        logical_bits |= UInt32(1) << UInt32(slot)
                    slot += 1
                mask_bits = logical_bits

        comptime for i in range(0, half_load):
            # rank1-style mask_r2p: turn bit into predicate and use it to select
            var bit: UInt32 = (mask_bits >> UInt32(i)) & UInt32(1)
            var in_bound: Bool = bit != UInt32(0)
            # masked_val = s_row[i]      if in_bound
            #            = MASK_VALUE    otherwise (finite sentinel; see
            #            module-level comment on MASK_VALUE for why)
            var val: Scalar[Self.AccumType] = s_row[i][0]
            var masked_val: Scalar[
                Self.AccumType
            ] = val if in_bound else MASK_VALUE

            comptime if NonCausalMask:
                var v: SIMD[Self.AccumType, 1] = masked_val
                var coord = clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos + cache_start_pos,
                    UInt32(col0 + i),
                    UInt32(tile_key_base),
                    num_keys,
                    cache_start_pos,
                )
                v = mask.mask(coord, v)
                masked_val = v[0]

            s_row[i][0] = masked_val
            current_max = max(current_max, masked_val)

        return current_max

    @staticmethod
    @always_inline
    def Softmax[
        native_fp8: Bool = False,
        num_sp_stages: Int = 2,
        fp8_p_stage_stride: Int = 0,
        has_per_token_scales: Bool = False,
        has_attn_sink: Bool = False,
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _op_has_variable_topk: Bool = False,
        # When True, per-row score_row decomposes via integer
        # division by num_q_heads to select the causal horizon per q_token.
        fold_q: Bool = False,
        # comptime number of q_tokens packed
        # into BM=64 under fold_q=True. Needed for the ragged LSE -inf fill
        # loop. Only consumed inside `comptime if fold_q` branches.
        q_len_fold: Int = 1,
    ](
        tmem_addr: UInt32,
        s_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        p_bars: DecodeSM100MiscMBars[
            num_stages=num_sp_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        p_smem_ptr: SharedMemPointer[Scalar[Self.q_type]],
        max_smem: SharedMemPointer[
            Scalar[Self.AccumType]
        ],  # 256x1 double-buffered
        li_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        out_smem: SharedMemPointer[Scalar[Self.output_dtype]],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_dtype, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _op_has_variable_topk,
        ],
        scale: Float32,
        mask: Self.MaskType,
        prompt_idx: UInt32,  # batch index
        lse_accum_split_ptr: Self.SplitAccumType,
        batch_size: Int,
        scale_k_smem: OptionalReg[SharedMemPointer[Float32]] = None,
        q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
        attn_sink_log2: Float32 = Float32(min_or_neg_inf[.float32]()),
        # Logical sparse indices and their valid per-q_token length, forwarded
        # to the inner apply_mask (documented there); required non-null when
        # SparseCausalLogical is derived below.
        logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
        logical_indices_stride: Int = 0,
        logical_indices_len: Int = 0,
    ):
        comptime MaskName: String = Self.MaskType.name()
        comptime MaskTypeName: String = Self.MaskType.get_type_name()
        comptime assert Self.AccumType.is_floating_point()

        comptime NoMask: Bool = (MaskName == "NullMask")
        comptime CausalMask: Bool = (MaskName == "CausalMask")
        # Sparse gather slots are ordered by indexer SCORE, not by logical
        # key position, so the fast slot-count causal mask (correct for
        # dense decode, where col_base IS the logical key index) is wrong
        # once topk selects a genuine subset of the causal prefix. Gated to
        # CausalMask only: sliding window is not supported on the sparse
        # kernels (enforced by their own comptime asserts).
        comptime SparseCausalLogical: Bool = _op_sparse and CausalMask
        # Sliding window: SlidingWindowCausalMask is causal + lower bound at
        # `causal_limit - window_size`. Detected via get_type_name (since
        # name() embeds the window value, e.g. "SlidingWindowCausalMask[64]").
        comptime SlidingWindowMask: Bool = (
            MaskTypeName == "SlidingWindowCausalMask"
        )
        # Window size: 0 if not sliding. The `MHAMask.sliding_window_size()`
        # trait method is the canonical channel for this — `Self.MaskType.
        # window_size` would fail type-checking inside the comptime if guard
        # because it's a struct parameter, not exposed on the trait.
        comptime _sliding_window_size: Int = Self.MaskType.sliding_window_size()

        # Same S base / stride as in mma()
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        # Double-buffered max SMEM: two 128-element buffers to eliminate the
        # race between the read at `lane_id ^ 64` and the next iteration's
        # write. Consecutive iterations alternate buffers so no extra barrier
        # is needed between the read and the following write.
        # Buffer selection uses branchless pointer arithmetic:
        #   buf_offset = (tiles_done & 1) * WARPGROUP_SIZE
        # yielding 0 for even iterations and WARPGROUP_SIZE for odd ones.
        comptime smem_1d_layout = tt_row_major[WARPGROUP_SIZE]()
        var li_Smem_Tensor = TileTensor[
            Self.AccumType, type_of(smem_1d_layout), address_space=.SHARED
        ](li_smem, smem_1d_layout)

        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        # For split-K: use num_keys_this_split for loop bounds
        # but keep num_keys (total) for masking with global KV positions
        var num_keys = offset_position.num_keys  # Total keys for masking
        var num_keys_this_split = (
            offset_position.num_keys_this_split
        )  # Keys for this split
        var kv_start_row = (
            offset_position.kv_start_row
        )  # Starting KV position for this split
        var cache_start_pos: UInt32 = 0
        # Logical cache length: equals cache_len() in dense mode and
        # whenever sparse topk >= actual_tokens; diverges only when sparse
        # topk genuinely subsets the causal prefix, which is exactly when
        # the causal horizon must be computed against logical (not slot)
        # positions. See cache_len_logical() docstring.
        var cache_len: Int = offset_position.cache_len_logical()
        var start_pos: UInt32 = offset_position.start_pos(cache_start_pos)

        # S consumer / P producer (N-stage wrappers, works for any num_sp_stages)
        var s_cons = DecodeSConsumerN[num_sp_stages](s_bars.consumer())
        var p_prod = DecodePProducerN[num_sp_stages](p_bars.producer())
        var c_prod = DecodeCProducer(c_bars.producer())
        var warp_idx = warp_id[broadcast=True]()
        # 0..127 inside the softmax WG
        var lane_id = thread_idx.x
        # Lane mapping inside the softmax warpgroup
        var row: Int = lane_id & 0x3F  # 0..63
        var half: Int = lane_id >> 6  # 0 or 1
        # Column range this thread owns in P
        var col0: Int = half * Self.config.BN_QK >> 1  # 0 or 32

        var q_head_idx: UInt32 = UInt32(block_idx.x) * UInt32(
            Self.config.BM
        ) + UInt32(row)
        # Per-row score_row under fold BM=64 packs q_len_fold *
        # num_q_heads rows; `row` identifies (q_local, head_local) and
        # apply_mask derives the causal horizon via row // num_q_heads.
        # Non-fold: single token per batch, block_idx.y is the q-token index.
        var score_row: UInt32
        comptime if fold_q:
            score_row = UInt32(row)
        else:
            score_row = UInt32(block_idx.y)

        var mi: Scalar[Self.AccumType] = min_or_neg_inf[Self.AccumType]()
        var li: Scalar[Self.AccumType] = 0.0
        comptime log2e_f32 = Scalar[Self.AccumType](log2e)
        comptime half_load = (Self.config.BN_QK >> 1)
        # ------------------------------------------------------------------
        # Fold sigma_Q (per-query-token scale) into scale_log2e.
        #
        # sigma_Q is per-token (varies by Q sequence position), but all
        # BM=64 rows in this CTA are different heads of the SAME Q token,
        # so sigma_Q is constant for the entire CTA. We fold it into
        # scale_log2e once (not per-KV-tile) so softmax scaling becomes:
        #   score * (scale * sigma_Q) * log2e
        #
        # The per-KV-token sigma_KV[t] is applied separately per column
        # (Place 1 for QK dequant, Place 2 for PV pre-fuse) and is
        # unchanged by this folding.
        # ------------------------------------------------------------------
        var effective_scale = scale
        comptime if has_per_token_scales:
            var _q_token_idx = offset_position.q_token_idx
            effective_scale = scale * q_scale_ptr.unsafe_value()[_q_token_idx]
        var scale_log2e = effective_scale.cast[Self.AccumType]()

        var tiles_done: Int = 0
        # Use num_keys_this_split for loop bounds (each split processes its portion)
        var num_k_tiles = ceildiv(num_keys_this_split, Self.config.BN_QK)
        # Sliding-window leading-tile skip + empty guard (comptime-gated;
        # entire block compiles away for non-sliding masks). MUST match the
        # producer (load/mmaQK/mmaPV) skip exactly so barrier counts agree.
        comptime if SlidingWindowMask:
            var _W_sw: Int = _sliding_window_size
            var _global_lo_sw = max(cache_len + 1 - _W_sw, 0)
            var _local_lo_sw = max(_global_lo_sw - kv_start_row, 0)
            var _tile_skip_sw = _local_lo_sw // Self.config.BN_QK
            # tiles_done starts at the skip count so apply_mask sees the
            # correct global key position via `kv_start_row + tiles_done * BN_QK`.
            tiles_done = _tile_skip_sw
            # Empty-split guard: split lies entirely below the window.
            if _tile_skip_sw >= num_k_tiles:
                num_k_tiles = _tile_skip_sw  # loop condition false
        # Index of the FIRST tile processed by this Softmax invocation.
        # Used to skip the c_prod.commit() on the very first tile (no prior
        # O accumulator to correct). For non-sliding masks this is 0,
        # recovering the original `tiles_done > 0` semantics. For sliding
        # window it equals `tiles_done`'s initial value above.
        var first_processed_tile_sw: Int = tiles_done
        var current_max: Scalar[Self.AccumType]
        while tiles_done < num_k_tiles:
            # Wait for an S slot to become ready
            var slot_idx: UInt32 = s_cons.wait()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            tcgen05_fence_after()

            # Each thread reads one full 32-element row (128 rows x 32 columns)
            var s_row = tt_stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[half_load]())
            var s_row_val = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=32,
                dtype=Self.AccumType,
                pack=False,
            ](s_tmem_slot)

            comptime for _i in range(type_of(s_row_val).length):
                s_row.raw_store(_i, s_row_val[_i])
            tcgen05_load_wait()

            s_cons.release()

            # ------------------------------------------------------------------
            # Per-token KV scale: load into registers ONCE per tile.
            #
            # When has_per_token_scales is True, each KV token t has a float32
            # scale sigma_KV[t] stored in scale SMEM. We need these scales in
            # TWO places: (1) QK dequant and (2) PV pre-fuse. To avoid
            # reading SMEM twice (2 x 32 x 4 = 256 bytes per place), we cache
            # all 32 sigma_KV values for this thread's columns into registers
            # ONCE here, then reuse them in both places.
            # ------------------------------------------------------------------
            # Register-cached per-token scales for this tile.
            # Declared outside the comptime if so it's in scope for Place 2.
            var _sigma_kv_regs = tt_stack_allocation[
                dtype=Self.AccumType, address_space=.LOCAL
            ](row_major[half_load]())
            comptime if has_per_token_scales:
                # Compute the scale SMEM pointer for this pipeline stage.
                # per_token_scales_per_stage bytes = BN_QK * 1 * sizeof(f32) = 256
                # In float32 elements per stage: BN_QK = 64.
                comptime _scale_elems_per_stage = Self.config.BN_QK
                var _scale_stage_ptr = (
                    scale_k_smem.unsafe_value()
                    + slot_idx * UInt32(_scale_elems_per_stage)
                )
                # Load all 32 sigma_KV values for this thread's columns into
                # registers ONCE. This is the ONLY SMEM read for scales in the
                # entire tile processing.
                # The last k-tile TMA may load OOB scale slots that
                # contain uninitialized NaN. After softmax, P=0 for those
                # columns, but 0*NaN=NaN poisons the PV MMA output. Clamping
                # via max(sigma, 0) maps NaN→0 per PTX semantics (max(NaN,0)=0)
                # and is a no-op for valid positive scales.
                comptime for _j in range(half_load):
                    _sigma_kv_regs.raw_store(
                        _j,
                        max(
                            rebind[Scalar[Self.AccumType]](
                                _scale_stage_ptr[col0 + _j]
                            ),
                            Scalar[Self.AccumType](0),
                        ),
                    )

                # Place 1: QK dequant — multiply each score column by its
                # token's sigma_KV[t] BEFORE the scale_log2e multiplication.
                # Uses register-cached scales (no SMEM read).
                # Fused with scale_log2e below when per-token scales active.

            var s_row_val_vectorized = s_row.vectorize[2]()
            comptime vs_count = ceildiv(half_load, 2)

            comptime if has_per_token_scales:
                # Fused Place 1 + scale_log2e: vectorized multiply of
                # sigma_KV and scale_log2e in a single pass to halve
                # instruction count vs two separate scalar loops.
                var _sigma_kv_vec_p1 = _sigma_kv_regs.vectorize[2]()
                comptime for _vi in range(vs_count):
                    s_row_val_vectorized[_vi] = (
                        s_row_val_vectorized[_vi]
                        * _sigma_kv_vec_p1[_vi]
                        * scale_log2e
                    )
            else:
                comptime for _vi in range(vs_count):
                    s_row_val_vectorized[_vi] = (
                        s_row_val_vectorized[_vi] * scale_log2e
                    )

            # under fold, pass num_q_heads so apply_mask's causal
            # branch can derive per-row horizon via `row // num_q_heads`.
            # fold_q_num_heads=0 (default)
            comptime _fold_q_num_heads: Int = (
                Self.config.num_q_heads if fold_q else 0
            )
            # Sliding window is causal + per-row lower bound; so the fast
            # path treats it as CausalMask=True with SlidingWindowSize set.
            comptime _causal_for_apply: Bool = CausalMask or SlidingWindowMask
            comptime if NoMask or CausalMask or SlidingWindowMask:
                current_max = Self.apply_mask[
                    half_load,
                    NonCausalMask=False,
                    CausalMask=_causal_for_apply,
                    fold_q_num_heads=_fold_q_num_heads,
                    SlidingWindowSize=_sliding_window_size,
                    SparseCausalLogical=SparseCausalLogical,
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,  # Pass kv_start_row for split-K global position
                    logical_indices=logical_indices,
                    logical_indices_stride=logical_indices_stride,
                    q_token_idx=offset_position.q_token_idx,
                    logical_indices_len=logical_indices_len,
                )
            else:
                current_max = Self.apply_mask[
                    half_load, NonCausalMask=True, CausalMask=False
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,  # Pass kv_start_row for split-K global position
                )
            current_max *= log2e_f32

            comptime rescale_threshold: Float32 = (
                Self.config.skip_correction_threshold
            )
            # Double-buffered write/read: even iterations use buffer 0,
            # odd iterations use buffer 1. Branchless selection via
            # (tiles_done & 1) * WARPGROUP_SIZE — one AND + one MUL + one ADD,
            # no divergent branch on the critical path.
            var buf_offset = (tiles_done & 1) * WARPGROUP_SIZE
            var max_buf = TileTensor[
                Self.AccumType, type_of(smem_1d_layout), address_space=.SHARED
            ](max_smem + buf_offset, smem_1d_layout)
            max_buf[lane_id] = current_max
            named_barrier[Int32(WARPGROUP_SIZE)](2)
            # 0 ^ 64 = 64, 1 ^ 64 = 65, ... 63 ^ 64 = 127
            # 64 ^ 64 = 0, 65 ^ 64 = 1, ... 127 ^ 64 = 63
            var other_half_max = max_buf[lane_id ^ 64][0]
            current_max = max(current_max, other_half_max)
            var new_max: Scalar[Self.AccumType] = max(mi, current_max)
            var diff = sub_ftz(rebind[Float32](mi), rebind[Float32](new_max))
            # `current_max` is initialized to
            # the finite MASK_VALUE in apply_mask, so `new_max >= MASK_VALUE`
            # (finite) on every iteration. First-iter `mi=-inf` gives
            # `diff = -inf - finite = -inf`, exp2(-inf)=0 (finite), no NaN.
            var scale_for_old_max: Scalar[Self.AccumType]
            # Per-lane predicate, not a warp vote: under `fold_q`, `row =
            # lane_id & 0x3F` packs `q_len_fold` distinct (never-committed
            # draft included) query tokens into one warp, so an OR here
            # would leak a sibling token's rescale trajectory into this
            # one's.
            if diff < rescale_threshold:
                scale_for_old_max = rebind[Scalar[Self.AccumType]](exp2(diff))
            else:
                scale_for_old_max = 1.0
                new_max = mi
            var float2_register = s_row.vectorize[2]()
            var float2_current_sum: SIMD[Self.AccumType, 2] = 0.0

            # With the finite MASK_VALUE in apply_mask, both `score` and
            # `new_max` are >= MASK_VALUE, so `score - new_max` is finite
            # (worst case `MASK_VALUE - MASK_VALUE = 0` for fully-masked rows,
            # giving `exp2(0) = 1` and `li = N`; the resulting partial_lse is
            # so negative that the combine kernel weights this split as 0).
            comptime for i in range(0, half_load // 2):
                var element = float2_register[i]
                float2_register[i] = exp2(element.fma(log2e_f32, -new_max))
                float2_current_sum += float2_register[i]

            # compute softmax using S_tmem_slot -> produce probabilities in regs
            # Expose correction scalars in SMEM for Correction warpgroup.
            # Skip the FIRST processed tile since there's no prior O
            # accumulator to correct. For non-sliding masks
            # `first_processed_tile_sw` is 0 (original `tiles_done > 0`).
            if tiles_done > first_processed_tile_sw:
                c_prod.acquire()
                # write back the exp2f(mi - new_max); to the correction_max_smem
                # corr_max_Smem_Tensor[lane_id] = scale_for_old
                # Issue the TMEM store: 32 datapaths × 32 bits × repeat=1
                var _scale_tuple = Array[Scalar[Self.AccumType], 1](
                    fill=scale_for_old_max
                )
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](corr_scale_tmem, _scale_tuple)
                tcgen05_store_wait()
                #  signal to the correction warpgroup:
                c_prod.commit()

            # wait until MMA has released P (consumer_mbar.phase matches)
            p_prod.acquire()
            var p_stage = p_prod.stage_index()

            # ------------------------------------------------------------------
            # Place 2: Per-token KV scale: pre-fuse sigma_KV[t] into P for
            # PV dequant. Uses register-cached scales loaded at the top of
            # the tile loop (no SMEM read).
            #
            # In MLA absorbed mode V derives from the same FP8 latent as K,
            # so it shares the same per-token scale sigma_KV[t]. The correct
            # PV output is: O[d] = sum_t P[t] * sigma_KV[t] * V_fp8[t][d].
            # We fuse sigma_KV[t] into P before it is written to SMEM and
            # consumed by the PV MMA: P'[t] = P[t] * sigma_KV[t].
            # ------------------------------------------------------------------
            comptime if has_per_token_scales:
                # Reuse register-cached sigma_KV values from Place 1.
                # Vectorized: use SIMD[Float32, 2] to halve instruction count.
                var _sigma_kv_vec_p2 = _sigma_kv_regs.vectorize[2]()
                var _s_row_vec_p2 = s_row.vectorize[2]()
                comptime _vs_count_p2 = ceildiv(half_load, 2)
                comptime for _vi in range(_vs_count_p2):
                    _s_row_vec_p2[_vi] = (
                        _s_row_vec_p2[_vi] * _sigma_kv_vec_p2[_vi]
                    )

            comptime if native_fp8:
                # FP8 path: P lives in SMEM (separate region or reusing rope)
                comptime fp8_p_type = DType.float8_e4m3fn
                # When fp8_p_stage_stride > 0, P reuses KV rope SMEM:
                # P_i is at rope_base + i * fp8_p_stage_stride (in FP8 elems).
                # When 0 (default), P stages are contiguous at BlockElems apart.
                comptime _p_stride = fp8_p_stage_stride if fp8_p_stage_stride > 0 else Self.BlockElems
                var p_smem_stage = p_smem_ptr.bitcast[
                    Scalar[fp8_p_type]
                ]() + p_stage * UInt32(_p_stride)
                write_fp8_row_to_smem_chunked[
                    half_load,
                    out_dtype=fp8_p_type,
                    in_dtype=Self.AccumType,
                    config=Self.config,
                ](p_smem_stage, s_row, col0, row)
            else:
                # BF16 path: P is embedded inside KV stage SMEM
                var p_smem = p_smem_ptr + (
                    p_stage * UInt32(Self.KVStageElems)
                    + UInt32(Self.NumVOBlocks * Self.BlockElems)
                )
                write_bf16x2_row_to_smem_chunked[
                    half_load,
                    out_dtype=Self.q_type,
                    in_dtype=Self.AccumType,
                    config=Self.config,
                ](p_smem, s_row, col0, row)

            fence_async_view_proxy()
            # 128 threads call -> producer_mbar.arrive() (128 arrivals) + state.step()
            p_prod.commit()
            mi = new_max
            li = li.fma(
                scale_for_old_max, float2_current_sum[0] + float2_current_sum[1]
            )
            # now update the li scale for the next tile
            tiles_done += 1

        li_Smem_Tensor[lane_id] = li
        named_barrier[Int32(WARPGROUP_SIZE)](2)
        li += li_Smem_Tensor[lane_id ^ 64][0]

        # --------------------------------------------------------------------------
        # Split-K: Store partial LSE to lse_accum_split for combine kernel
        # --------------------------------------------------------------------------
        # LSE (Log-Sum-Exp) in log2 format: lse = log2(li) + mi
        # This allows the combine kernel to merge partial results:
        #   global_lse = log2(sum(exp2(lse_i - max_lse))) + max_lse
        #   scale_i = exp2(lse_i - global_lse)
        #   final_output = sum(scale_i * partial_output_i)
        #
        # LSE accumulator shape: (num_splits, batch_size, seq_len, num_heads)
        # Strides: stride_split = batch_size * seq_len * num_heads
        #          stride_batch = seq_len * num_heads
        #          stride_seq = num_heads
        comptime if Self.config.decoding_warp_split_k:
            # Only threads with valid heads should write LSE
            # head_idx = block_idx.x * BM + row (where row is 0-63 for each half)
            # Each thread in the warpgroup handles one row (one head)
            # row = lane_id & 0x3F gives 0-63 for both halves
            # half = lane_id >> 6 gives 0 or 1
            # We only need one write per head, so half=0 threads write
            var half_idx = lane_id >> 6  # 0 for first half, 1 for second half

            # Under fold, BM=64 packs q_len_fold * num_q_heads rows
            # and THIS CTA owns all q_max_seq_len LSE slots for the batch. So
            # rows where q_local >= seq_len (ragged) must explicitly write
            # -inf into their LSE slots (the accum buffer is uninitialized
            # for those slots, unlike the non-fold path where ragged CTAs
            # take pdl_early_exit and never reach Softmax).
            comptime if fold_q:
                var q_local = row // Self.config.num_q_heads
                var head_local = row % Self.config.num_q_heads
                if half_idx == 0 and row < (
                    q_len_fold * Self.config.num_q_heads
                ):
                    var partial_lse: Scalar[Self.AccumType]
                    if q_local >= offset_position.seq_len:
                        partial_lse = min_or_neg_inf[Self.AccumType]()
                    else:
                        partial_lse = (
                            log2(max(li[0], Scalar[Self.AccumType](0))) + mi
                        )
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads
                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + q_local * stride_seq
                        + head_local
                    )
                    var lse_ptr = rebind[
                        UnsafePointer[
                            Scalar[Self.AccumType], origin=MutAnyOrigin
                        ]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[lse_offset] = partial_lse
            else:
                var head_idx = block_idx.x * Self.config.BM + row
                if half_idx == 0 and head_idx < Self.config.num_q_heads:
                    # Compute LSE in log2 format: log2(li) + mi
                    # li is the running sum of exp2 values; mi is the running max
                    # in log2 scale. When all scores in this split are causally
                    # masked, the online softmax produces NaN via exp2(-inf+inf),
                    # poisoning li. Clamping li to 0 makes log2(0)=-inf, and
                    # -inf + mi(-inf) = -inf, giving this split zero weight in
                    # the combine kernel (same as pdl_early_exit for empty splits).
                    # On NVIDIA GPUs, max(NaN, 0) = 0 per PTX semantics.
                    var partial_lse = (
                        log2(max(li[0], Scalar[Self.AccumType](0))) + mi
                    )

                    # LSE offset calculation:
                    # lse_accum_split shape: (num_splits, batch_size, max_seq_len, num_heads)
                    # Use max_seq_len (not per-batch seq_len) for strides to match
                    # the PADDED buffer layout and the combine kernel's read pattern.
                    var seq_idx = block_idx.y
                    var stride_batch = (
                        offset_position.max_seq_len * Self.config.num_q_heads
                    )
                    var stride_split = batch_size * stride_batch
                    var stride_seq = Self.config.num_q_heads

                    var lse_offset = (
                        offset_position.split_idx * stride_split
                        + offset_position.batch_idx * stride_batch
                        + seq_idx * stride_seq
                        + head_idx
                    )
                    # need to rebind the pointer to mutable pointer for write access
                    var lse_ptr = rebind[
                        UnsafePointer[
                            Scalar[Self.AccumType], origin=MutAnyOrigin
                        ]
                    ](lse_accum_split_ptr.value())
                    lse_ptr[lse_offset] = partial_lse

        # --------------------------------------------------------------------------
        # Epilogue: scale output by recip(li) and write to shared memory as bf16
        # --------------------------------------------------------------------------
        comptime assert (
            Self.AccumType == .float32
        ), "accumulator type should be float32"
        comptime assert (
            Self.output_dtype == .bfloat16
        ), "output type should be bfloat16"

        comptime DecodeOutProducerType = DecodeOutProducer[
            Self.output_dtype, Self.config
        ]
        comptime blocks_per_stage = DecodeOutProducerType.blocks_per_stage
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)

        # By the time we reach to epilogue the KV is free.
        # So we can safely use the KV buffer for writing the output and have more async write.
        # however the write function massively changes based on the mma_n size
        # so when mma_n is 256 we have the first 128 columns with warp0/1 and the
        # next 128 column with warp2/3 tiles and so on for the next 256 columns
        # it is 256/32 which is equivalent of 512/64

        # Half-stripe width in fp32 per warp_pair = BN_PV/8.
        comptime epi_half_load: UInt32 = UInt32((Self.config.BN_PV // 4) >> 1)
        comptime chunk_size: Int = 16
        comptime total_elems: Int = Int(epi_half_load) * blocks_per_stage
        var out_prod = DecodeOutProducer[Self.output_dtype, Self.config](
            out_pipeline, out_smem
        )

        # Pre-compute scale factor.
        # Guard against NaN in li (possible when all scores in a split are
        # masked, producing exp2(-inf+inf)=NaN that poisons li). Using
        # `li[0] > 0` instead of `li[0] != 0` ensures NaN maps to 0,
        # zeroing the output for this split — consistent with the LSE path's
        # max(li, 0) guard and the combine kernel's weighting.
        #
        # Attn sink (no-split only): account for non-selected tokens by
        # adding exp2(attn_sink_log2 - mi) to the denominator. For the
        # split path, attn_sink is deferred to the combine kernel to
        # avoid double-counting across splits.
        var o_scale_li: Scalar[Self.AccumType]
        comptime if has_attn_sink and not Self.config.decoding_warp_split_k:
            # No-split path with attn_sink: o_scale = 1 / (li + exp2(attn_sink_log2 - mi))
            # FlashMLA reference: kernel.cuh:346
            var denominator = li[0] + exp2(
                attn_sink_log2.cast[Self.AccumType]() - mi
            )
            o_scale_li = (
                recip(SIMD[Self.AccumType, 1](denominator))[0] if li[0]
                > 0 else 0
            )
        else:
            # Split path or no attn_sink: standard recip(li).
            # FlashMLA reference: kernel.cuh:387
            o_scale_li = recip(li)[0] if li[0] > 0 else 0

        var warp_pair = UInt32(warp_idx >> 1)
        var epi_col0: Int = Int(
            warp_pair * epi_half_load * UInt32((blocks_per_stage >> 1) ^ 1)
        )

        # Number of MMA PV rounds (outer loop) and iterations within each round (inner loop)
        # MMA_PV_N=256 processes 4 blocks (256/64=4) at a time
        # depth=512 has 8 blocks total, so 2 MMA PV rounds (512/256=2)
        # Each round has (MMA_PV_N/BN_QK)/blocks_per_stage = (256/64)/2 = 2 iterations
        # corr_done_bars has 2 slots matching the 2 MMA PV rounds
        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3
        # The pattern repeats every MMA_PV_N (256) columns
        comptime num_mma_pv_rounds = Self.config.depth // Self.config.MMA_PV_N
        # iters_per_mma_round = (MMA_PV_N / (BN_PV/4)) / blocks_per_stage = 4 / 2.
        comptime iters_per_mma_round = 4 // blocks_per_stage

        comptime for mma_round in range(num_mma_pv_rounds):
            # Wait for Correction to finish corrections for this MMA PV round
            corr_done_bars.mbar_base[mma_round].wait(0)

            # Fence to ensure all MMA writes to O TMEM are visible before we read
            tcgen05_fence_after()

            comptime for slot in range(iters_per_mma_round):
                # Global iteration index combining mma_round and slot
                comptime i = mma_round * iters_per_mma_round + slot

                var o_tmem_base: UInt32 = o_tmem + UInt32(
                    i
                ) * epi_half_load * UInt32(blocks_per_stage)

                # Load all data for this tile into a LocalTensor
                var o_row_subtile = tt_stack_allocation[
                    dtype=Self.AccumType, address_space=.LOCAL
                ](row_major[total_elems]())
                var _o_ld_result = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=total_elems,
                    dtype=Self.AccumType,
                    pack=False,
                ](o_tmem_base)

                comptime for _i in range(total_elems):
                    o_row_subtile.raw_store(_i, _o_ld_result[_i])
                tcgen05_load_wait()

                out_prod.acquire()
                var stage_ptr = out_prod.stage_base_ptr(
                    Int(warp_pair * UInt32(blocks_per_stage >> 1))
                )

                # Write O to shared memory with scaling
                write_bf16x2_row_to_smem_chunked[
                    total_elems,
                    out_dtype=Self.output_dtype,
                    in_dtype=Self.AccumType,
                    config=Self.config,
                    chunk_size=chunk_size,
                    scale_needed=True,
                ](stage_ptr, o_row_subtile, epi_col0, row, o_scale_li)

                out_prod.commit_step()

    # --------------------------------------------------------------------------
    # MLA decoding Correction kernel
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def Correction[
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _op_has_variable_topk: Bool = False,
    ](
        tmem_addr: UInt32,
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _op_has_variable_topk,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        var o_cons = DecodeOConsumer(o_bars.consumer())
        var c_cons = DecodeCConsumer(c_bars.consumer())
        var tiles_done: Int = 1

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Sliding-window leading-tile skip — comptime-gated; entire block
        # compiles away for non-sliding masks. Correction starts AFTER
        # Softmax's first processed tile, i.e. at `tile_skip + 1`. Must
        # match the load skip exactly so producer/consumer iterations align.
        # Empty-split (tile_skip >= num_k_tiles) cannot reach here in
        # split-K mode because the kernel-level pdl_early_exit fires first.
        comptime _sliding_window_mask_corr: Bool = (
            Self.MaskType.get_type_name() == "SlidingWindowCausalMask"
        )
        comptime if _sliding_window_mask_corr:
            comptime _W_corr: Int = Self.MaskType.sliding_window_size()
            var _global_lo_corr = max(
                offset_position.cache_len() + 1 - _W_corr, 0
            )
            var _local_lo_corr = max(
                _global_lo_corr - offset_position.kv_start_row, 0
            )
            var _tile_skip_corr = _local_lo_corr // Self.config.BN_QK
            tiles_done = _tile_skip_corr + 1

        while tiles_done < num_k_tiles:
            # after computing per-row c_scalar from max/li:
            c_cons.wait()
            # 2) Issue TMEM load: 32 datapaths × 32 bits × repeat=1
            var scale_value_tuple = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=1,
                dtype=Self.AccumType,
                pack=False,
            ](corr_scale_tmem)
            tcgen05_load_wait()
            c_cons.release()
            var scale_value = scale_value_tuple[0]
            var change = _vote_nvidia_helper(scale_value < 1.0) != 0
            comptime num_o_tiles = Self.config.MMA_PV_N // (
                Self.output_tile_width * 2
            )
            comptime o_range = Self.config.depth // Self.config.MMA_PV_N
            # the MMA.ws split the output across two warps
            comptime o_stride = Self.config.MMA_PV_N // 2

            comptime for slot_idx in range(o_range):
                o_cons.wait()
                if change:
                    comptime for i in range(0, num_o_tiles):
                        # Here we load from o_tmem. it is 32 bit float and we load 64 fp32 element per tile
                        var o_tmem_subtile: UInt32 = (
                            o_tmem
                            + UInt32(i) * UInt32(Self.config.BN_QK)
                            + UInt32(slot_idx) * UInt32(o_stride)
                        )
                        var o_row_subtile = tt_stack_allocation[
                            dtype=Self.AccumType,
                            address_space=.LOCAL,
                        ](row_major[Self.config.BN_QK]())
                        var _o_ld_corr = tcgen05_ld[
                            datapaths=32,
                            bits=32,
                            repeat=Self.config.BN_QK,
                            dtype=Self.AccumType,
                            pack=False,
                        ](o_tmem_subtile)

                        comptime for _i in range(Self.config.BN_QK):
                            o_row_subtile.raw_store(_i, _o_ld_corr[_i])
                        tcgen05_load_wait()

                        var float2_register = o_row_subtile.vectorize[2]()

                        comptime for j in range(0, Self.config.BN_QK // 2):
                            var element = rebind[SIMD[Self.AccumType, 2]](
                                float2_register[j]
                            )
                            float2_register[j] = rebind[
                                type_of(float2_register[j])
                            ](element * SIMD[Self.AccumType, 2](scale_value))
                        var _o_st_corr = Array[_, Self.config.BN_QK](
                            fill_with_unrolled=lambda [i: Int]() -> Scalar[
                                Self.AccumType
                            ]: o_row_subtile.raw_load(i)
                        )
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat=Self.config.BN_QK,
                            pack=False,
                        ](
                            o_tmem_subtile,
                            _o_st_corr,
                        )
                        tcgen05_store_wait()
                o_cons.release()
            tiles_done += 1

        # Wait on the final O from MMA before signaling Softmax
        o_cons.wait()
        # Signal to Softmax that first 4 blocks are ready (slot 0)
        _ = corr_done_bars.mbar_base[0].arrive()
        o_cons.release()
        # second stage of the correction pipeline
        o_cons.wait()
        # Signal to Softmax that all corrections are done and O is ready (slot 1)
        _ = corr_done_bars.mbar_base[1].arrive()
        # Release the final O barrier
        o_cons.release()

    # --------------------------------------------------------------------------
    # MLA decoding store kernel
    # --------------------------------------------------------------------------
    # If it goes to the batch loop remember correction is out of sync with MMA on
    # O_tmem wait and release as it starts one stage before MMA
    @staticmethod
    @always_inline
    def store[
        _op_sparse: Bool = False,
        _op_has_extra_kv: Bool = False,
        _op_has_variable_topk: Bool = False,
        # When True, the fold caller strides the output-store
        # TMA per-q_token via a dedicated out_row_offset_at(q_local) accessor.
        fold_q: Bool = False,
        # comptime number of q_tokens packed
        # into BM=64 under fold_q=True. Only consumed inside the
        # `comptime if fold_q` branch; when fold_q=False, ignored.
        q_len_fold: Int = 1,
    ](
        out_pipeline: OutPipeline[
            num_out_stages=DecodeOutProducer[
                Self.output_dtype, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        out_smem: SharedMemPointer[Scalar[Self.output_dtype]],
        o_tma: ORaggedTMATile[
            dtype=Self.output_dtype,
            BM=Self.config.out_rows,
            # BF16/SWIZZLE_128B clamps innermost to 64 (= BN_PV/4).
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            _op_sparse,
            _op_has_extra_kv,
            _op_has_variable_topk,
        ],
    ):
        comptime DecodeOutConsumerType = DecodeOutConsumer[
            Self.output_dtype, Self.config
        ]
        comptime col_per_warp = DecodeOutConsumerType.col_per_warp
        comptime blocks_per_stage = DecodeOutConsumerType.blocks_per_stage
        comptime num_out_stages = DecodeOutConsumerType.num_out_stages
        comptime num_out_stages_per_mma = num_out_stages // num_mma_pv
        comptime num_mma_pv = Self.config.padded_depth // Self.config.MMA_PV_N
        var out_cons = DecodeOutConsumer[Self.output_dtype, Self.config](
            out_pipeline, out_smem
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.out_row_offset
        var rows_to_store = rows_owned[Self.config](Int(block_idx.x))

        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3

        comptime for n in range(0, num_mma_pv):
            comptime for m in range(0, num_out_stages_per_mma):
                out_cons.wait()

                comptime for k in range(0, blocks_per_stage):
                    var stage_ptr = out_cons.stage_base_ptr(k)
                    var col: Int = (
                        n * Self.config.MMA_PV_N
                        + m * (Self.config.BN_PV // 4)
                        + k * col_per_warp
                    )
                    comptime o_elements = (
                        Self.config.out_rows * (Self.config.BN_PV // 4)
                    )
                    comptime o_tt_layout = tt_row_major[o_elements]()
                    comptime if fold_q:
                        # Fold: BM=64 TMEM packs q_len_fold * num_q_heads;
                        # emit one TMA store per q_token.
                        comptime for q_local in range(q_len_fold):
                            var q_stage_ptr = stage_ptr + (
                                q_local
                                * Self.config.num_q_heads
                                * (Self.config.BN_PV // 4)
                            )
                            var smem_tensor = TileTensor[
                                Self.output_dtype,
                                type_of(o_tt_layout),
                                address_space=.SHARED,
                            ](q_stage_ptr, o_tt_layout)
                            if is_leader:
                                fence_async_view_proxy()
                                o_tma.async_store_3d(
                                    smem_tensor,
                                    store_row_coords[Self.config.out_rows](
                                        col,
                                        offset_position.out_row_offset_at(
                                            q_local
                                        ),
                                        rows_to_store,
                                    ),
                                )
                    else:
                        var smem_tensor = TileTensor[
                            Self.output_dtype,
                            type_of(o_tt_layout),
                            address_space=.SHARED,
                        ](stage_ptr, o_tt_layout)
                        if is_leader:
                            fence_async_view_proxy()
                            o_tma.async_store_3d(
                                smem_tensor,
                                store_row_coords[Self.config.out_rows](
                                    col, row, rows_to_store
                                ),
                            )
                out_cons.release(elect_mask)
        if is_leader:
            o_tma.commit_group()
        o_tma.wait_group[0]()
