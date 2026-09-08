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
"""SM100 (B200) sparse MLA prefill shared surface (BF16 + FP8 KV variants).

Holds the config, base shared-memory layout, QK/SV MMA operand structs, and
`MLAPrefillSparseCommon` — the dtype-agnostic machinery (Q-load prologue, the
QK/SV MMA pipeline `mma` with its `fp8_active` flag, `cp_q_from_smem_to_tmem`,
`_raw_indices_to_tma_rows`, `kv_valid_producer`) instantiated by both the
BF16-KV kernel (`mla_prefill_sparse.mojo`) and the FP8-KV kernel
(`mla_prefill_sparse_kv_fp8.mojo`).  Mirrors the sparse-decode split
(`mla_decode_utils.mojo` + `mla_decode_sparse_kv_{bf16,fp8}.mojo`).
"""

from std.sys import size_of
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    warp_id,
    thread_idx,
    WARP_SIZE,
)
from max.gpu.sync import barrier
from std.math import ceildiv, exp2
from std.math.constants import log2e
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.primitives.cluster import cluster_sync
import max.gpu.primitives.warp as warp
from max.gpu.memory import (
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_mbarrier_init,
    fence_async_view_proxy,
)
from max.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host import DeviceContext, FuncAttribute
from std.ffi import UnsafeUnion
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_cp,
    tcgen05_store_wait,
    tcgen05_fence_after,
    tcgen05_fence_before,
)

from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from nn.attention.mha_mask import MHAMask
from kv_cache.types import KVCacheT
from nn.attention.mha_utils import OptionallyStaticInt, MHAPartitionScheme
from nn.attention.gpu.nvidia.common import (
    elect,
    OptionalPointer,
    NullPointer,
    NonNullPointer,
)

from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.mha import q_num_matrix_view_rows

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    add_ftz,
    sub_ftz,
    mul_ftz,
    fma_ftz,
    exp2_emulation,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from linalg.arch.sm100.mma import smem_descriptor


from layout import (
    TileTensor,
    row_major,
    Idx,
    TensorLayout,
    Coord,
    stack_allocation as tt_stack_allocation,
)
from layout.swizzle import make_swizzle
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    ld_shared_v4_u32,
    cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4,
    st_shared_v4_b32_at_bf16_elem_off,
    hmul2_bf16x8_by_scalar,
)
from std.memory import bitcast
from layout.tma_async import (
    create_tensor_tile,
    TMATensorTile,
    SharedMemBarrier,
    RaggedTMA3DTile,
    _gather4_box_width,
    _default_desc_shape,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.utils.numerics import min_or_neg_inf


struct MLASparseConfig[
    qkv_dtype: DType,
    b_topk_: Int = 128,
    num_mbars_: Int = 2,
    q_smem_depth_: Int = 192,
    q_tmem_depth_: Int = 384,
    cta_group_: Int = 2,
]:
    var num_q_heads: Int
    # Padded per-problem head count = the MMA M-tile. `tcgen05.mma kind::f16`
    # only permits M in {64, 128} (1-CTA / 2-CTA); head counts < 64 are illegal
    # as an M-tile, so they run a padded 64-row tile (cta_group=1). This value
    # drives every MMA / SMEM / descriptor geometry that must span the padded
    # tile; `num_q_heads` stays the *real* head count for Q-load, O-store and
    # the output view. NFC at 64 (-> 64) and 128 (-> 128).
    var padded_num_q_heads: Int
    var num_kv_heads: Int
    var qk_depth: Int
    # Width of the Q / KV row as the TMA finds it in global memory. Equal to
    # `qk_depth` for the rotary-carrying row the SMEM/TMEM split is laid out
    # for. A NoPE model stores a narrower row -- latent only, no rotary tail --
    # and the kernel zero-fills the difference, so this drives the TMA
    # descriptors and the transaction byte count and nothing else. It must
    # never reach the SMEM/TMEM layout or the MMA width.
    var input_qk_depth: Int
    var v_depth: Int
    var indices_stride: Int
    var group: Int

    # the leftmost q_depth is store in smem,
    # the rightmost q_depth is store in tmem
    # for the leftmost qk_depth mma, we do ss_mma,
    # for the rightmost qk_depth mma, we do ts_mma,
    comptime cta_group = Self.cta_group_
    comptime use_ws = Self.cta_group_ == 1
    comptime q_smem_depth = Self.q_smem_depth_
    comptime q_tmem_depth = Self.q_tmem_depth_
    comptime B_TOPK = Self.b_topk_
    comptime num_mbars = Self.num_mbars_
    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime num_threads: Int = 512
    comptime sm100_tmem_cols = 512

    comptime q_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime k_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

    @always_inline
    def __init__(
        out self,
        *,
        num_q_heads: Int,
        num_kv_heads: Int,
        qk_depth: Int,
        v_depth: Int,
        indices_stride: Int,
        group: Int,
        # 0 means "use `qk_depth`" (every rotary-carrying call site).
        input_qk_depth: Int = 0,
    ):
        self.num_q_heads = num_q_heads
        self.padded_num_q_heads = 64 if num_q_heads <= 64 else num_q_heads
        self.num_kv_heads = num_kv_heads
        self.qk_depth = qk_depth
        self.input_qk_depth = input_qk_depth if input_qk_depth > 0 else qk_depth
        self.v_depth = v_depth
        self.indices_stride = indices_stride
        self.group = group


struct MLASparseSharedMemory[config: MLASparseConfig]:
    comptime num_q_heads = Self.config.num_q_heads
    comptime qkv_dtype = Self.config.qkv_dtype
    comptime qk_depth = Self.config.qk_depth
    comptime num_mbars = Self.config.num_mbars

    # per cta dimension
    comptime BH = Self.config.padded_num_q_heads // Self.config.cta_group
    comptime TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    # Per-CTA TMEM cluster N (= MMA_N for one SV atom). Each CTA holds the
    # full cluster N output for its M slice (M-split, N-shared on output).
    comptime NUM_SV_ATOMS = 2
    comptime SV_ATOM_MMA_N = Self.config.v_depth // Self.NUM_SV_ATOMS
    comptime V_DEPTH_PER_CTA = Self.SV_ATOM_MMA_N
    comptime V_BMN_PER_ATOM = Self.SV_ATOM_MMA_N // Self.config.cta_group
    comptime O_ATOM_PHYS_COLS = Self.SV_ATOM_MMA_N // 2
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS

    comptime FULL_Q_SIZE = Self.BH * Self.qk_depth
    # split num_q_heads per cta
    comptime SHARED_Q_SIZE = Self.BH * Self.config.q_smem_depth
    # split b_topk per cta
    comptime K_SIZE = Self.TOPK_PER_CTA * Self.config.qk_depth
    # V smem: B_TOPK rows × V_SMEM_COLS_PER_CTA cols per CTA.
    comptime V_SIZE = Self.config.B_TOPK * Self.V_SMEM_COLS_PER_CTA
    comptime SHARED_QKV_SIZE = Self.SHARED_Q_SIZE + Self.K_SIZE + Self.V_SIZE
    comptime S_SIZE = Self.BH * Self.config.B_TOPK
    comptime O_SIZE = Self.BH * Self.config.v_depth

    comptime FULL_Q_TYPE = Array[Scalar[Self.qkv_dtype], Self.FULL_Q_SIZE]
    comptime SHARED_QKV_TYPE = Array[
        Scalar[Self.qkv_dtype], Self.SHARED_QKV_SIZE
    ]
    comptime O_TYPE = Array[Scalar[Self.qkv_dtype], Self.O_SIZE]

    var qkvo_union: UnsafeUnion[
        Self.FULL_Q_TYPE, Self.SHARED_QKV_TYPE, Self.O_TYPE
    ]
    var scores: Array[Scalar[Self.qkv_dtype], Self.S_SIZE]
    var p: Array[Float32, Self.S_SIZE]

    var prologue_q: Array[SharedMemBarrier, 1]
    var prologue_q_cp: Array[SharedMemBarrier, 1]

    # use for qk ss_mma completion (used by qk_ss_mma)
    var qk_ss_done: Array[SharedMemBarrier, Self.num_mbars]
    # use for qk ts_mma completion (used by qk_ts_mma)
    var qk_ts_done: Array[SharedMemBarrier, Self.num_mbars]

    # use for sv_p0 completion
    var sv_p0_done: Array[SharedMemBarrier, Self.num_mbars]
    # use for sv_p1 completion
    var sv_p1_done: Array[SharedMemBarrier, Self.num_mbars]

    # use for k_p0 ready TMA load completion (used by k_p0)
    var k_p0_ready: Array[SharedMemBarrier, Self.num_mbars]
    # use for k_p1 ready TMA load completion (used by k_p1)
    var k_p1_ready: Array[SharedMemBarrier, Self.num_mbars]

    # use for v_p0 ready TMA load completion (used by v_p0)
    var v_p0_ready: Array[SharedMemBarrier, Self.num_mbars]
    # use for v_p1 ready TMA load completion (used by v_p1)
    var v_p1_ready: Array[SharedMemBarrier, Self.num_mbars]

    var p_free: Array[SharedMemBarrier, Self.num_mbars]
    var so_ready: Array[SharedMemBarrier, Self.num_mbars]

    # k_valid_ready / k_valid_free coordinate the WG3 warp-13 producer
    # that writes the per-position validity bitmask (`is_k_valid` below)
    # consumed by WG0's mask step.  Mirrors phase1.cuh's `bar_k_valid_*`
    # + `is_k_valid` slot.
    var k_valid_ready: Array[SharedMemBarrier, Self.num_mbars]
    var k_valid_free: Array[SharedMemBarrier, Self.num_mbars]
    # Validity bitmask: 1 bit per topk position.  Packed as
    # MASK_BYTES_PER_BUF = B_TOPK / 8 bytes per buffer; byte `j` of buffer
    # `b` carries bits for positions [j*8 .. j*8+8).  Bit set = "this index
    # is in range AND its absolute position is < topk_lengths[seq]".
    # Stored flat (no nested Array) so the byte-offset math at
    # producer/consumer relies on simple array layout, not on the
    # absence of inter-element padding in nested Array.
    comptime MASK_BYTES_PER_BUF = Self.config.B_TOPK // 8
    # Producer parallelism: one lane per output mask byte; each lane
    # packs INDICES_PER_LANE = 8 bits.  Coupled by definition.
    comptime NUM_KV_VALID_LANES = Self.MASK_BYTES_PER_BUF
    comptime INDICES_PER_LANE = 8
    var is_k_valid: Array[UInt8, Self.num_mbars * Self.MASK_BYTES_PER_BUF]
    var tmem_addr: Array[UInt32, 1]

    # store rowwise max and sum for each threads in a warp group
    var rowwise_max: Array[Float32, WARPGROUP_SIZE]
    var rowwise_sum: Array[Float32, WARPGROUP_SIZE]


struct QKMMAOp[dtype: DType, accum_dtype: DType, config: MLASparseConfig]:
    comptime SSMMAType = SM100TensorAccumulator[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.padded_num_q_heads,
        MMA_N=Self.config.B_TOPK,
        BK=Self.config.q_smem_depth,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F16,
        cta_group=Self.config.cta_group,
    ]
    # 3 stages: BK=384 with MMA_K=16 needs 24 k-mmas, which exceeds the
    # bulk_mma per-instruction limit of 16. Splitting into 3 stages of 8
    # k-mmas each keeps every issued instruction inside the limit. Each
    # `.mma[stage_idx=i]` call issues its share; stages 1+ always
    # accumulate (c_scale forced to 1 internally).
    comptime TSMMAType = SM100TensorAccumulator[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.padded_num_q_heads,
        MMA_N=Self.config.B_TOPK,
        BK=Self.config.q_tmem_depth,
        a_tmem=True,
        mma_kind=UMMAKind.KIND_F16,
        cta_group=Self.config.cta_group,
        num_stages=3,
    ]
    comptime NUM_TS_STAGES = 3

    @staticmethod
    @always_inline
    def smem_descriptor_q(
        q_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.padded_num_q_heads // Self.config.cta_group,
            BK=Self.config.q_smem_depth,
            swizzle_mode=Self.config.q_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    def tmem_descriptor_q(
        q_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.padded_num_q_heads // Self.config.cta_group,
            BK=Self.config.q_tmem_depth,
            swizzle_mode=Self.config.q_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    def descriptor_k_p0(
        k_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.B_TOPK // Self.config.cta_group,
            BK=Self.config.q_smem_depth,
            swizzle_mode=Self.config.k_swizzle_mode,
            is_k_major=True,
        ](k_smem)

    @staticmethod
    @always_inline
    def descriptor_k_p1(
        k_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.B_TOPK // Self.config.cta_group,
            BK=Self.config.q_tmem_depth,
            swizzle_mode=Self.config.k_swizzle_mode,
            is_k_major=True,
        ](k_smem)


struct SVMMAType[dtype: DType, accum_dtype: DType, config: MLASparseConfig]:
    # MMA_N is per-atom cluster N.  The UMMA descriptor encodes N>>3 in 6
    # bits (max value 504), so a single atom can't cover MMA_N=512.  We
    # split SV into NUM_SV_ATOMS=2 atoms of MMA_N=v_depth/cta_group=256
    # each (matching phase1.cuh's `SM100_MMA_F16BF16_2x1SM_SS_NOELECT<...,
    # 256, ...>`).  The caller issues both atoms per SV iter into disjoint
    # O TMEM regions (O_TMEM_ADDR and O_TMEM_ADDR_ATOM2).
    # A (S) is written flat by WG0 → swizzle_a=NONE. B (V) is loaded via
    # gather4 with SWIZZLE_128B, producing col-group-blocked smem the
    # SW128 MMA descriptor reads correctly (same pattern as
    # `DecodeSM100PVSS.descriptor_v_block`).
    comptime SS_P0MMAType = SM100TensorAccumulator[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.padded_num_q_heads,
        MMA_N=Self.config.v_depth // 2,
        BK=Self.config.B_TOPK // 2,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=TensorMapSwizzle.SWIZZLE_NONE,
        swizzle_b=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=False,
        cta_group=Self.config.cta_group,
    ]
    comptime SS_P1MMAType = SM100TensorAccumulator[
        Self.dtype,
        Self.accum_dtype,
        MMA_M=Self.config.padded_num_q_heads,
        MMA_N=Self.config.v_depth // 2,
        BK=Self.config.B_TOPK // 2,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F16,
        swizzle_a=TensorMapSwizzle.SWIZZLE_NONE,
        swizzle_b=TensorMapSwizzle.SWIZZLE_128B,
        transpose_b=False,
        cta_group=Self.config.cta_group,
    ]

    # S smem is written by WG0 with a flat (non-swizzled) layout — see the
    # store loop in WG0's per-block softmax that writes `s_bf16` into
    # `scores_ptr` at uint128 strides matching phase1.cuh:166-167. To make
    # the MMA's read interpretation match, the descriptor must declare
    # SWIZZLE_NONE — phase1.cuh uses `Layout_K_INTER_Atom<bf16>` (the
    # "INTER" = interleave-only = non-swizzled atom) for the same reason.
    # Previously this descriptor used SWIZZLE_128B; with our flat writes,
    # the MMA was reading cols permuted by the swizzle XOR and producing
    # wrong O values.
    @staticmethod
    @always_inline
    def descriptor_s(
        s_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.padded_num_q_heads // Self.config.cta_group,
            BK=Self.config.B_TOPK // 2,  # 64 columns
            swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
            is_k_major=True,
        ](s_smem)

    # V smem is loaded via gather4 with `swizzle_mode=SWIZZLE_128B` and
    # `tile_width=V_SMEM_COLS_PER_CTA` (= 256 = 2 atoms × b_bmn).  gather4
    # splits into 4 col-groups (each 64 cols × B_TOPK rows) at smem stride
    # B_TOPK*64.  BMN here = V_BMN_PER_ATOM = MMA_N/cta_group = 128 matches
    # one SV atom's per-CTA b_bmn (atom1 reads smem cols 0..127, atom2
    # reads cols 128..255 with a shifted base pointer).
    @staticmethod
    @always_inline
    def descriptor_v(
        v_smem: UnsafePointer[Scalar[Self.dtype], address_space=.SHARED, ...],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=(Self.config.v_depth // 2) // Self.config.cta_group,
            BK=Self.config.B_TOPK // 2,  # 64 K rows per part
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            is_k_major=False,
        ](v_smem)


struct MLAPrefillSparseCommon[
    KVLUTType: MHAOperand,
    output_dtype: DType,
    config: MLASparseConfig,
](TrivialRegisterPassable):
    comptime qkv_dtype = Self.config.qkv_dtype
    comptime accum_dtype = DType.float32

    comptime q_smem_depth = Self.config.q_smem_depth
    comptime q_tmem_depth = Self.config.q_tmem_depth
    comptime qkv_dtype_size = size_of[Self.qkv_dtype]()

    comptime NUM_Q_HEADS_PER_CTA = Self.config.num_q_heads // Self.config.cta_group
    # Padded per-CTA head rows = the MMA M-tile per CTA (always 64: 64/1 at
    # head<=64, 128/2 at head128). Use this for geometry that must span the
    # full padded MMA tile (O TMEM read stride, S-scratch p1 offset, cp_q Q-row
    # stride); use NUM_Q_HEADS_PER_CTA for the real rows to load/store. Equal at
    # 64/128 (NFC); differ only when num_q_heads < 64.
    comptime PADDED_HEADS_PER_CTA = (
        Self.config.padded_num_q_heads // Self.config.cta_group
    )
    comptime B_TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    comptime NUM_SV_ATOMS = 2
    comptime SV_ATOM_MMA_N = Self.config.v_depth // Self.NUM_SV_ATOMS
    comptime V_DEPTH_PER_CTA = Self.SV_ATOM_MMA_N
    comptime V_BMN_PER_ATOM = Self.SV_ATOM_MMA_N // Self.config.cta_group
    comptime O_ATOM_PHYS_COLS = Self.SV_ATOM_MMA_N // 2
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS

    # The Q TMA box requests the REAL per-CTA head count so every sub-copy
    # reads only in-bounds GMEM heads (a padded box makes heads
    # [num_q_heads, 64) a middle-dimension OOB region whose TMA never
    # completes its mbarrier transaction -> deadlock). For num_q_heads < 64
    # the depth-tile sub-copies are issued manually so each real-head box
    # lands at the PADDED (64-row) depth-tile stride the BMN=64 QK-MMA reads
    # (see the Q-load in `kernel`). NFC at 64/128 (real == padded).
    comptime q_tile_shape = Index(
        1, Self.NUM_Q_HEADS_PER_CTA, Self.config.input_qk_depth
    )
    comptime q_desc_shape = _default_desc_shape[
        3, Self.qkv_dtype, Self.q_tile_shape, Self.config.q_swizzle_mode
    ]()

    comptime k_tile_width = Self.config.input_qk_depth
    comptime k_swizzle_mode = Self.config.k_swizzle_mode
    comptime k_tile_height = Self.B_TOPK_PER_CTA
    comptime k_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.k_tile_width, Self.k_swizzle_mode
    ]()
    comptime k_tile_shape = Index(Self.k_tile_height, Self.k_gather_box)
    comptime k_desc_shape = Index(1, Self.k_gather_box)

    comptime v_tile_width = Self.V_SMEM_COLS_PER_CTA
    # V uses SWIZZLE_128B; gather4 splits tile_width=128 into 2 col-groups
    # of 64 bf16 each (box_width = 128B / sizeof(bf16) = 64).
    comptime v_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    # tile_height is HALF of B_TOPK: each call to async_copy_gather4_tile
    # loads one K-half (64 rows × V_SMEM_COLS_PER_CTA cols), and load_v
    # invokes it twice with different smem bases. With tile_height=64 the
    # gather4 col-group stride (tile_height*box_w = 4096 elems) matches
    # the SW128 BMN=128 MN-major descriptor's mn_outer stride.
    comptime v_tile_height = Self.config.B_TOPK // 2
    comptime v_gather_box = _gather4_box_width[
        Self.qkv_dtype, Self.v_tile_width, Self.v_swizzle_mode
    ]()
    comptime v_tile_shape = Index(Self.v_tile_height, Self.v_gather_box)
    comptime v_desc_shape = Index(1, Self.v_gather_box)

    # desc_shape inner dim 64×2=128B satisfies the ≤256B SWIZZLE_NONE constraint.
    # SMEM is written in column-group order to match TMA's sub-copy layout.
    comptime o_tile_shape = Index(Self.NUM_Q_HEADS_PER_CTA, Self.config.v_depth)
    comptime o_desc_shape = Index(Self.NUM_Q_HEADS_PER_CTA, 64)

    # FP8 TMA swizzle modes. SWIZZLE_NONE paired with INT64 packing means one
    # gather4 descriptor covers the full row without inner col-tiling.
    comptime FP8_K_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE
    comptime FP8_V_SWIZZLE = TensorMapSwizzle.SWIZZLE_NONE

    # FP8 K TMA: INT64-packed, one descriptor covers 576 FP8 bytes/row.
    # BF16 path uses k_tile_shape/k_desc_shape; FP8 path uses these.
    # Kept separate because `DType.int64 if fp8 else qkv_dtype` trips a
    # Mojo parser bug in comptime positions.
    comptime k_tma_dtype_fp8 = DType.int64
    comptime k_tma_tile_width_fp8 = Self.config.input_qk_depth // 8
    comptime k_tma_swizzle_fp8 = Self.FP8_K_SWIZZLE
    comptime k_tma_gather_box_fp8 = _gather4_box_width[
        Self.k_tma_dtype_fp8,
        Self.k_tma_tile_width_fp8,
        Self.k_tma_swizzle_fp8,
    ]()
    comptime k_tma_tile_shape_fp8 = Index(
        Self.k_tile_height, Self.k_tma_gather_box_fp8
    )
    comptime k_tma_desc_shape_fp8 = Index(1, Self.k_tma_gather_box_fp8)

    # FP8 V TMA: INT64-packed, per-atom width (V_BMN_PER_ATOM / 8 INT64 elems).
    # Two gather4 calls per row group (one per SV atom) because the atoms'
    # gmem columns are not contiguous.
    comptime v_tma_dtype_fp8 = DType.int64
    comptime v_tma_tile_width_fp8 = Self.V_BMN_PER_ATOM // 8
    comptime v_tma_swizzle_fp8 = Self.FP8_V_SWIZZLE
    comptime v_tma_gather_box_fp8 = _gather4_box_width[
        Self.v_tma_dtype_fp8,
        Self.v_tma_tile_width_fp8,
        Self.v_tma_swizzle_fp8,
    ]()
    # FP8 V loads all B_TOPK rows at once (no key-half split like BF16).
    comptime v_tma_tile_height_fp8 = Self.config.B_TOPK
    comptime v_tma_tile_shape_fp8 = Index(
        Self.v_tma_tile_height_fp8, Self.v_tma_gather_box_fp8
    )
    comptime v_tma_desc_shape_fp8 = Index(1, Self.v_tma_gather_box_fp8)

    comptime SMemType = MLASparseSharedMemory[Self.config]
    comptime FULL_Q_TYPE = Self.SMemType.FULL_Q_TYPE
    comptime SHARED_QKV_TYPE = Self.SMemType.SHARED_QKV_TYPE
    comptime O_TYPE = Self.SMemType.O_TYPE

    comptime QKMMAOpType = QKMMAOp[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]
    comptime SVMMAType = SVMMAType[
        Self.qkv_dtype, Self.accum_dtype, Self.config
    ]

    comptime O_TMEM_ADDR = 0
    # atom2's O sits right after atom1's, offset by the physical footprint
    # O_ATOM_PHYS_COLS (not the V-operand width V_BMN_PER_ATOM).
    comptime O_TMEM_ADDR_ATOM2 = Self.O_TMEM_ADDR + Self.O_ATOM_PHYS_COLS
    comptime P_TMEM_ADDR = 256
    comptime Q_TMEM_ADDR = 512 - Self.q_tmem_depth // 2

    # tcgen05.commit multicast mask for MMA-completion barriers: signal both
    # CTAs in the pair (0b11) at cta_group=2, only self (0b1) at cta_group=1.
    comptime CTA_MASK: UInt16 = 0b11 if Self.config.cta_group == 2 else 0b1

    @always_inline
    @staticmethod
    def _load_q_prologue(
        full_q_ptr: UnsafePointer[
            mut=True, Scalar[Self.config.qkv_dtype], address_space=.SHARED, ...
        ],
        q_tma_op: TMATensorTile[
            Self.qkv_dtype,
            3,
            Self.q_tile_shape,
            Self.q_desc_shape,
        ],
        prologue_q_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        cta_id: UInt32,
        seq_idx: UInt32,
    ):
        """Q-load prologue shared by `kernel` (BF16) and `kernel_fp8`.

        Q is BF16 in BOTH the BF16 and FP8 KV-cache paths, so the Q load is
        byte-identical; factoring it here keeps the sub-64-head memset +
        manual depth-tile sub-copy fix in ONE place so the two kernels can't
        drift.  Must be called by ALL threads (the padded-Q memset + barrier
        are warp-uniform); only warp 0 issues the TMA.  NFC at 64/128 (real ==
        padded -> no memset, plain built-in async_copy).
        """
        # Sub-64-head configs run a padded 64-row (PADDED_HEADS_PER_CTA) QK-MMA
        # M-tile, so `full_q` SMEM is a 64-row k-major SWIZZLE_128B tile, but
        # only `num_q_heads` real rows are loaded. Zero the ENTIRE padded Q
        # buffer first so the padding rows [num_q_heads, 64) feed finite (0)
        # scores into the padded MMA rows (never uninitialized SMEM, which
        # could be NaN and poison the warp-uniform softmax votes). A full-buffer
        # memset is swizzle-invariant; the sub-copies below overwrite only the
        # real rows. `fence_async_view_proxy` publishes the generic-proxy zero
        # stores to the async proxy so the tcgen05 Q read (SS-MMA + smem->tmem
        # cp) sees them. Guarded to num_q_heads < 64, so 64/128 stay byte-NFC.
        comptime if Self.NUM_Q_HEADS_PER_CTA < Self.PADDED_HEADS_PER_CTA:
            for i in range(
                Int(thread_idx.x),
                Self.SMemType.FULL_Q_SIZE,
                Self.config.num_threads,
            ):
                full_q_ptr[i] = Scalar[Self.config.qkv_dtype](0)
            barrier()
            fence_async_view_proxy()

        if warp_id() == 0:
            if elect_one_sync():
                # The TMA coord on the head dim is in ELEMENTS, not tiles
                # — passing `cta_id` (0 or 1) made CTA 1 load heads
                # `1..64` instead of `64..127`, shifting CTA 1's Q in
                # smem by -63 heads.  The Q_TMEM dump confirmed this:
                # h=33 (CTA 0 local 33) was duplicated into CTA 1's
                # local row 32 (= position of h=33 in heads 1..64).
                comptime if (
                    Self.NUM_Q_HEADS_PER_CTA < Self.PADDED_HEADS_PER_CTA
                ):
                    # Manual depth-tile sub-copies. Each reads a REAL-head
                    # box [1, num_q_heads, Q_SWIZZLE_COLS] from GMEM (fully
                    # in-bounds -> the mbarrier transaction completes, no
                    # hang) but writes it to `full_q` at the PADDED depth-tile
                    # stride (PADDED_HEADS_PER_CTA * Q_SWIZZLE_COLS) the BMN=64
                    # QK-MMA / smem->tmem cp expect. Mirrors async_copy_3d's
                    # cp_async_bulk_tensor_shared_cluster_global issue but with
                    # the padded (not naive real*cols) offset. SWIZZLE_128B has
                    # an 8-row core matrix, so a real box (num_q_heads a
                    # multiple of 8, asserted below) lands rows
                    # [0, num_q_heads) at the same swizzled positions a 64-row
                    # box would; the untouched rows stay zeroed above.
                    comptime Q_SWIZZLE_COLS = Self.q_desc_shape[2]
                    comptime NUM_Q_DEPTH_TILES = (
                        Self.config.input_qk_depth // Q_SWIZZLE_COLS
                    )
                    comptime Q_PADDED_DEPTH_TILE_STRIDE = (
                        Self.PADDED_HEADS_PER_CTA * Q_SWIZZLE_COLS
                    )
                    comptime for j in range(NUM_Q_DEPTH_TILES):
                        cp_async_bulk_tensor_shared_cluster_global[
                            cta_group=Self.config.cta_group
                        ](
                            full_q_ptr + j * Q_PADDED_DEPTH_TILE_STRIDE,
                            UnsafePointer(to=q_tma_op.descriptor).bitcast[
                                NoneType
                            ](),
                            prologue_q_ptr[].unsafe_ptr(),
                            Index(
                                j * Q_SWIZZLE_COLS,
                                Int(cta_id) * Self.NUM_Q_HEADS_PER_CTA,
                                Int(seq_idx),
                            ),
                        )
                else:
                    # REAL-head Q SMEM view matching the real-head
                    # q_tile_shape box (real == padded on the 64/128 path).
                    var q_full_smem_tensor = TileTensor(
                        full_q_ptr,
                        row_major[
                            1,
                            Self.NUM_Q_HEADS_PER_CTA,
                            Self.config.input_qk_depth,
                        ](),
                    )
                    q_tma_op.async_copy[cta_group=Self.config.cta_group](
                        q_full_smem_tensor,
                        prologue_q_ptr[],
                        StaticTuple[UInt32, 3](
                            0,
                            cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA),
                            seq_idx,
                        ),
                    )

    @always_inline
    @staticmethod
    def mma[
        # Under FP8 the K/V producer warpgroups credit k_p*_ready /
        # v_p*_ready via a manual arrive() after the FP8->BF16 convert,
        # not via TMA complete_transaction. Skip the expect_bytes calls
        # below so the consumer wait conditions don't block on an
        # expect_tx that no one will decrement.
        fp8_active: Bool = False,
    ](
        q_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        k_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        s_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        v_smem_ptr: UnsafePointer[
            Scalar[Self.qkv_dtype], address_space=.SHARED, ...
        ],
        k_p0_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k_p1_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        v_p0_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        v_p1_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        sv_p0_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        sv_p1_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        so_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        p_free: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        qk_ss_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        qk_ts_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k: UInt32,
        num_k_blocks: UInt32,
    ):
        # The caller runs this method under an `elect_one_sync()` guard (a single
        # lane of warp 12), so `elect()` returns 1 here. Forward it to every MMA
        # below instead of a hard-coded `1`, keeping the predicate
        # `elect.sync`-derived (and self-protecting if that guard is widened).
        var e = elect()
        if k < num_k_blocks:
            # QK^T MMA
            # wait for k load p0
            var cur_buf = k % UInt32(Self.SMemType.num_mbars)
            var cur_phase = k / UInt32(Self.SMemType.num_mbars) & 1
            var prev_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
            var prev_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1

            comptime if not fp8_active:
                k_p0_ready[cur_buf].expect_bytes(
                    Int32(
                        Self.config.B_TOPK
                        * Self.config.q_smem_depth
                        * Self.qkv_dtype_size
                    )
                )
            k_p0_ready[cur_buf].wait(cur_phase)
            if k > 0:
                # Wait for WG0 (the consumer of P) to release the prev P
                # buffer before letting the MMA overwrite it via the next
                # QK GEMM. Previously this incorrectly waited on
                # k_p0_ready[prev_buf], which is the producer barrier for K
                # and a separate state — that race was latent because WG0
                # never arrived on p_free in the original stub.
                p_free[prev_buf].wait(prev_phase)

            tcgen05_fence_after()
            var q_smem_desc = Self.QKMMAOpType.smem_descriptor_q(q_smem_ptr)
            var k_p0_smem_desc = Self.QKMMAOpType.descriptor_k_p0(k_smem_ptr)
            # c_scale=0 mirrors phase1.cuh:539 `utcmma_ss(..., true)` — the
            # first k-mma of the QK SS gemm clears P_TMEM (D = A@B) rather
            # than accumulating onto stale tmem.
            Self.QKMMAOpType.SSMMAType.mma(
                q_smem_desc,
                k_p0_smem_desc,
                Self.P_TMEM_ADDR,
                c_scale=0,
                elect=e,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                qk_ss_done[cur_buf].unsafe_ptr(),
                Self.CTA_MASK,
            )

            # wait for k load p1
            comptime if not fp8_active:
                k_p1_ready[cur_buf].expect_bytes(
                    Int32(
                        Self.config.B_TOPK
                        * (
                            Self.config.input_qk_depth
                            - Self.config.q_smem_depth
                        )
                        * Self.qkv_dtype_size
                    )
                )
            k_p1_ready[cur_buf].wait(cur_phase)
            tcgen05_fence_after()

            var k_p1_smem_desc = Self.QKMMAOpType.descriptor_k_p1(
                k_smem_ptr + Self.B_TOPK_PER_CTA * Self.config.q_smem_depth
            )
            # TS MMA is split across NUM_TS_STAGES stages (see
            # TSMMAType definition).  Each stage adds its k-batch to P;
            # stage 0 uses the passed c_scale (here 1 = accumulate onto
            # SS's P), stages 1+ force c_scale=1 internally.
            comptime for stage_idx in range(Self.QKMMAOpType.NUM_TS_STAGES):
                Self.QKMMAOpType.TSMMAType.mma[stage_idx=stage_idx](
                    UInt32(Self.Q_TMEM_ADDR),
                    k_p1_smem_desc,
                    Self.P_TMEM_ADDR,
                    c_scale=1,
                    elect=e,
                )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                qk_ts_done[cur_buf].unsafe_ptr(),
                Self.CTA_MASK,
            )
        if k > 0:
            # O += S(i-1)V(i-1)
            var curr_buf = (k - 1) % UInt32(Self.SMemType.num_mbars)
            var cur_phase = (k - 1) / UInt32(Self.SMemType.num_mbars) & 1

            # SV descriptors for the 2 atoms × 2 key-halves = 4 sub-MMAs.
            # Atom1 reads V smem cols 0..127 (cluster depths 0..255), atom2
            # reads cols 128..255 (cluster depths 256..511).  Within a
            # key-half, atom2's base is shifted by B_TOPK/2 *
            # V_BMN_PER_ATOM elements (= 8192 for our dims) — that's the
            # 2-cg sub-tile boundary the v loader produces (cgs 0,1 =
            # atom1; cgs 2,3 = atom2 within each key-half).
            comptime ATOM2_COL_OFFSET = (
                Self.config.B_TOPK // 2 * Self.V_BMN_PER_ATOM
            )
            comptime KEY_HALF_OFFSET = (
                Self.config.B_TOPK // 2 * Self.V_SMEM_COLS_PER_CTA
            )
            var s_p0_smem_desc = Self.SVMMAType.descriptor_s(s_smem_ptr)
            var s_p1_smem_desc = Self.SVMMAType.descriptor_s(
                s_smem_ptr + Self.PADDED_HEADS_PER_CTA * Self.config.B_TOPK // 2
            )
            var v_atom1_p0_desc = Self.SVMMAType.descriptor_v(v_smem_ptr)
            var v_atom1_p1_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + KEY_HALF_OFFSET
            )
            var v_atom2_p0_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + ATOM2_COL_OFFSET
            )
            var v_atom2_p1_desc = Self.SVMMAType.descriptor_v(
                v_smem_ptr + KEY_HALF_OFFSET + ATOM2_COL_OFFSET
            )

            so_ready[curr_buf].wait(cur_phase)

            # Cluster total bytes = cta_group * per-CTA bytes for ONE
            # key-half = cta_group * (B_TOPK/2 * V_SMEM_COLS_PER_CTA *
            # sizeof) = (B_TOPK/2) * (V_DEPTH_PER_CTA * cta_group) *
            # sizeof = (B_TOPK/2) * v_depth * sizeof.  V_SMEM_COLS_PER_CTA
            # now holds both atoms' cols, so the cluster bytes correspond
            # to one key-half of *both* atoms together.
            comptime if not fp8_active:
                v_p0_ready[curr_buf].expect_bytes(
                    Int32(
                        Self.config.B_TOPK
                        // 2
                        * Self.config.v_depth
                        * Self.qkv_dtype_size
                    )
                )
            v_p0_ready[curr_buf].wait(cur_phase)
            # Mirrors phase1.cuh:565 — TMEM fence after the v_part0 wait
            # ensures the SV P0 MMA's read of O_TMEM (under c_scale=1 for
            # k>1) sees the rescaled O written by the softmax warpgroup,
            # not stale TMEM.  Bar_so_ready only orders smem traffic
            # (fence_async_view_proxy), not TMEM.
            tcgen05_fence_after()

            # 2 SV atoms × 2 key-halves = 4 SS_MMA calls per SV iter.
            # accum_init (c_scale=0) is per-atom on the first SV iter
            # (k==1); subsequent iters accumulate.  Both atoms init
            # independently because they write to disjoint O TMEM regions.
            var sv_p0_c_scale: UInt32 = 0 if k == 1 else 1
            Self.SVMMAType.SS_P0MMAType.mma(
                s_p0_smem_desc,
                v_atom1_p0_desc,
                Self.O_TMEM_ADDR,
                c_scale=sv_p0_c_scale,
                elect=e,
            )
            Self.SVMMAType.SS_P0MMAType.mma(
                s_p0_smem_desc,
                v_atom2_p0_desc,
                UInt32(Self.O_TMEM_ADDR_ATOM2),
                c_scale=sv_p0_c_scale,
                elect=e,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                sv_p0_done[curr_buf].unsafe_ptr(),
                Self.CTA_MASK,
            )

            comptime if not fp8_active:
                v_p1_ready[curr_buf].expect_bytes(
                    Int32(
                        Self.config.B_TOPK
                        // 2
                        * Self.config.v_depth
                        * Self.qkv_dtype_size
                    )
                )
            v_p1_ready[curr_buf].wait(cur_phase)
            tcgen05_fence_after()

            # SV P1 always accumulates onto O (phase1.cuh:574,
            # `accum_init=false`).
            Self.SVMMAType.SS_P1MMAType.mma(
                s_p1_smem_desc,
                v_atom1_p1_desc,
                Self.O_TMEM_ADDR,
                c_scale=1,
                elect=e,
            )
            Self.SVMMAType.SS_P1MMAType.mma(
                s_p1_smem_desc,
                v_atom2_p1_desc,
                UInt32(Self.O_TMEM_ADDR_ATOM2),
                c_scale=1,
                elect=e,
            )
            mma_arrive_multicast[cta_group=Self.config.cta_group](
                sv_p1_done[curr_buf].unsafe_ptr(),
                Self.CTA_MASK,
            )

    @always_inline
    @staticmethod
    def cp_q_from_smem_to_tmem(
        smem_desc: MMASmemDescriptorPair,
        tmem_addr: UInt32,
    ):
        # each cta holds 64 x (q_smem_depth + 384)
        # we do 64x128bit tcgen05_cp
        # break down 384 to 6 64 col tiles, each tcgen05_cp copies 64x8
        # so we are essentially doing 64x(6x8x8)
        comptime NUM_Q_CP_TILES = Self.q_tmem_depth // 64
        comptime NUM_SUB_TILES = 64 // 8
        comptime for tile_id in range(NUM_Q_CP_TILES):
            comptime for sub_tile_id in range(NUM_SUB_TILES):
                # tile_id stride is *Q-row* stride (num head rows per
                # CTA × 64 col atom), NOT K-row stride.  B_TOPK_PER_CTA
                # happens to equal NUM_Q_HEADS_PER_CTA in the current
                # config (both = 64) but the semantically correct name is
                # the Q-side one.
                comptime sub_tile_offset = (
                    tile_id * Self.PADDED_HEADS_PER_CTA * 64 + sub_tile_id * 8
                ) * Self.qkv_dtype_size
                comptime TMEM_OFFSET = tile_id * 32 + sub_tile_id * 4
                var sub_tile_desc = smem_desc + UInt32(sub_tile_offset)
                # Multicast pattern matches phase1.cuh
                # `SM100_UTCCP_2x64dp128bitlw0213_2cta::copy` (the "0213"
                # suffix = warpx2::02_13).  The cp_q tile/sub-tile loop
                # math also mirrors phase1.cuh's: tile_idx stride =
                # 8192 bytes (= 64 rows × 64 BF16 × 2), sub_tile stride
                # = 16 bytes (= 8 BF16 × 2).
                tcgen05_cp[
                    cta_group=Int32(Self.config.cta_group),
                    datapaths=64,
                    bits=128,
                    multicast="warpx2::02_13",
                ](
                    tmem_addr + UInt32(TMEM_OFFSET),
                    sub_tile_desc.descriptor(),
                )

    @always_inline
    @staticmethod
    def _raw_indices_to_tma_rows(
        kv_lut: Self.KVLUTType, raw: SIMD[.int32, 4]
    ) -> SIMD[.int32, 4]:
        """Map raw sparse `indices` entries to physical gather4 TMA rows.

        The sparse `indices` buffer stores encoded positions
        `physical_block * page_size + offset`.  The gather4 descriptors here
        are built with a global row stride of a single KV row
        (`tile_stride = qk_depth`), so each gather row index must be run
        through `kv_lut.get_tma_row`, which folds in the paged block stride
        `_stride() = num_layers * page_size` rows per physical block.  At
        `num_layers == 1` this is the identity; for `num_layers > 1` skipping
        it drops the `num_layers` factor and the gather lands in the wrong
        layer's cache region (progressively worse as the physical block index
        grows).  Mirrors the decode path's `_transform_indices_to_smem`
        (`mla_decode_sparse_kv_bf16.mojo`), including preserving the `-1`
        padding sentinel so invalid lanes stay invalid.
        """
        var rows = SIMD[.int32, 4]()
        comptime for i in range(4):
            rows[i] = raw[i] if raw[i] < Int32(0) else kv_lut.get_tma_row(
                raw[i]
            )
        return rows

    @always_inline
    @staticmethod
    def kv_valid_producer(
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        is_k_valid_ptr: UnsafePointer[
            mut=True, UInt8, address_space=.SHARED, ...
        ],
        k_valid_ready_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k_valid_free_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        lane_idx: UInt32,
        indices_base: UInt32,
        num_kv_rows: Int32,
        top_k_length: Int32,
        num_k_blocks: Int,
    ):
        # NUM_KV_VALID_LANES active lanes × INDICES_PER_LANE indices
        # = B_TOPK indices/k-block.  The bit-pack matches
        # phase1.cuh:583-598's `is_ks_valid_mask`.
        comptime INDICES_PER_LANE = Self.SMemType.INDICES_PER_LANE
        comptime MASK_BYTES_PER_BUF = Self.SMemType.MASK_BYTES_PER_BUF
        for k_block in range(num_k_blocks):
            var cur_buf = UInt32(k_block) % UInt32(Self.SMemType.num_mbars)
            # WG0 starts in phase 0 waiting for k_valid_ready; warp 13
            # is the producer, so it waits on k_valid_free with the
            # XOR-flipped phase (initial wait returns immediately since
            # the bar is fresh).  Matches phase1.cuh's
            # `wait((k/NUM_BUFS)&1^1)`.
            var free_phase = (
                UInt32(k_block) // UInt32(Self.SMemType.num_mbars)
            ) & UInt32(1) ^ UInt32(1)

            # Issue the gmem indices load + mask compute BEFORE waiting on
            # k_valid_free, so the producer overlaps gmem latency with the
            # consumer's prior iteration.  The result sits in registers
            # across the wait.
            var gidx_offset = (
                indices_base
                + UInt32(k_block) * UInt32(Self.config.B_TOPK)
                + lane_idx * UInt32(INDICES_PER_LANE)
            )
            # Sentinel-by-design: `indices` is uint32 in gmem; the cast to
            # int32 here is what makes the padding sentinel `0xFFFFFFFF`
            # alias to `-1` and fail the `idx_i >= 0` check below.  Assumes
            # `num_kv_rows` fits in signed int32 (~2B rows); far above any
            # realistic deployment.
            var idx_v8 = indices.load[width=INDICES_PER_LANE](
                Coord(gidx_offset)
            ).cast[.int32]()

            var abs_pos_base = Int32(k_block) * Int32(
                Self.config.B_TOPK
            ) + Int32(lane_idx) * Int32(INDICES_PER_LANE)
            var mask: UInt8 = 0
            comptime for i in range(INDICES_PER_LANE):
                var idx_i = idx_v8[i]
                var abs_pos = abs_pos_base + Int32(i)
                if (
                    idx_i >= Int32(0)
                    and idx_i < num_kv_rows
                    and abs_pos < top_k_length
                ):
                    mask = mask | (UInt8(1) << UInt8(i))

            k_valid_free_ptr[Int(cur_buf)].wait(free_phase)
            is_k_valid_ptr[
                Int(cur_buf) * MASK_BYTES_PER_BUF + Int(lane_idx)
            ] = mask
            _ = k_valid_ready_ptr[Int(cur_buf)].arrive()
