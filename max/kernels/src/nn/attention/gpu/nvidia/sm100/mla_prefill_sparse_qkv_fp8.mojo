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
"""SM100 (B200) native-FP8 sparse MLA prefill kernel.

Two comptime-selected tiles share this file: head<=64 on a single CTA
(cta_group=1) with K and V sharing one k-major gather buffer, and head=128 on a
2-CTA cluster (cta_group=2) with CTA-split K + a separate full-V gather (the
shared-KV simplification is impossible when CTA-split-K@base0 and full-V@base0
collide).  The cg2 cluster scaffold (grid, cta_id/seq_idx split, tcgen05_alloc
[cta_group], CTA_MASK multicast, cross-CTA arrive_cluster handshakes) mirrors the
dequant `mla_prefill_sparse_kv_fp8.mojo`; the compute stays native f8f6f4.  Both
tiles read V mn-major from a k-major SW64 buffer at BK=b_topk=64.

All of Q, K, V, and P are FP8 e4m3; QK^T and P*V run natively on tensor cores
via `tcgen05.mma.kind::f8f6f4` (KIND_F8F6F4) with NO in-SMEM FP8->BF16 dequant.
This removes the `convert_k/v_fp8_to_bf16` stage (and its LOCAL register staging)
of `mla_prefill_sparse_kv_fp8.mojo`, so the kernel spills zero registers by
construction and holds K/V at 1 byte/element.

Structure (top-k gather, 4-warpgroup layout, launch) mirrors
`mla_prefill_sparse_kv_fp8.mojo`.  Native-FP8 MMA machinery (single SS QK^T,
2-atom P*V with V read mn-major from the same k-major KV buffer, FP8 P written to
a separate SMEM region) mirrors the decode reference `mla_decode_qkv_fp8.mojo`
(`DecodeSM100QKTSS_FP8` / `DecodeSM100PVSS_FP8`).  Both operand-layout
foundations were validated bit-exact on B200; see
`.claude/agent-memory/mojo-kernel-engineer/fp8-sw64-gather4-qk-mma-validated.md`.

SMEM layout (all FP8 operands, num_mbars=2 stages):
  Q FP8:      64 x 576 x 1 = 36864 bytes            (persistent, SWIZZLE_64B)
  KV FP8:     2 x 64 x 576 x 1 = 73728 bytes        (double-buffered, SWIZZLE_64B)
  P FP8:      2 x 64 x 64 x 1 = 8192 bytes           (double-buffered, SWIZZLE_64B)
  O bf16:     reuses the KV region at the epilogue

The KV buffer is the single k-major SW64 gather; V is read mn-major from its
first `v_depth` columns (K and V share the MLA latent, so there is no separate V
gather).  Scale handling targets the DSv3.2 production `scale_block_size == 0`
(unit-scale) path: kv_scale == v_scale == 1, so `scale_log2e` and `output_scale`
match the BF16-KV kernel's unit-scale formulas.  Tensorwise/blockwise scale-fold
and head=128 (cta_group=2, 2-CTA f8f6f4) are follow-ups.
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
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
    tcgen05_fence_after,
    tcgen05_fence_before,
)
from std.memory import bitcast
from std.utils.numerics import min_or_neg_inf

from nn.attention.mha_operand import MHAOperand, KVCacheMHAOperand
from kv_cache.types import KVCacheT
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.common import elect
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    add_ftz,
    mul_ftz,
    st_shared_v4_b32,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAKind,
    mma_arrive_multicast,
)
from linalg.arch.sm100.mma import smem_descriptor

from layout import (
    TileTensor,
    row_major,
    Idx,
    TensorLayout,
    TensorEngine,
    Coord,
)
from layout.swizzle import make_swizzle, make_ldmatrix_swizzle
from layout.tma_async import (
    create_tensor_tile,
    TMATensorTile,
    SharedMemBarrier,
    _gather4_box_width,
    _default_desc_shape,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle

from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLAPrefillSparseCommon,
    MLASparseConfig,
)


comptime FP8_TYPE = DType.float8_e4m3fn
comptime SW64 = TensorMapSwizzle.SWIZZLE_64B


struct MLASparseSharedMemoryQKVFP8[config: MLASparseConfig]:
    """Native-FP8 SMEM layout: FP8 Q/KV/P + FP8-agnostic softmax scratch.

    Unlike `MLASparseSharedMemoryFP8`, there is no BF16 K/V region and no FP8
    staging (the operand IS the FP8 data the MMA reads), so K/V halve in bytes
    and the LOCAL dequant staging arrays disappear entirely.
    """

    comptime num_mbars = Self.config.num_mbars
    comptime PADDED_HEADS = Self.config.padded_num_q_heads
    # Per-CTA head rows = the MMA M-tile per CTA (64: 64/1 at head<=64, 128/2 at
    # head=128). Every SMEM region that holds one CTA's slice sizes off this.
    comptime PADDED_HEADS_PER_CTA = Self.config.padded_num_q_heads // Self.config.cta_group
    comptime NUM_Q_HEADS = Self.config.num_q_heads
    comptime B_TOPK = Self.config.B_TOPK
    # K is CTA-split (each CTA gathers its B_TOPK/cta_group rows); V is not.
    comptime B_TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    comptime qk_depth = Self.config.qk_depth
    comptime v_depth = Self.config.v_depth
    comptime is_cg2 = Self.config.cta_group == 2
    # Per-CTA V depth-slice width (each CTA holds v_depth/cta_group columns).
    comptime V_SMEM_COLS_PER_CTA = Self.v_depth // Self.config.cta_group

    comptime Q_SIZE = Self.PADDED_HEADS_PER_CTA * Self.qk_depth
    # K buffer.  At cg1 K and V share this k-major SW64 gather; at cg2 K is
    # CTA-split here and V lives in the separate `v` buffer below.
    comptime KV_STAGE_SIZE = Self.B_TOPK_PER_CTA * Self.qk_depth
    # Separate V buffer (cg2 only; the shared-KV simplification is impossible at
    # cta_group=2 where K@base0 and full-V@base0 would collide).  At cg1 it is a
    # 128-byte-multiple dummy so the following (`p`) MMA-operand buffer keeps its
    # 128B alignment (all TMA/MMA SMEM buffers are aligned by 128B-multiple size).
    comptime V_STAGE_SIZE = (
        Self.B_TOPK * Self.V_SMEM_COLS_PER_CTA
    ) if Self.is_cg2 else 128
    comptime P_STAGE_SIZE = Self.PADDED_HEADS_PER_CTA * Self.B_TOPK
    comptime O_SIZE = Self.PADDED_HEADS_PER_CTA * Self.v_depth
    # Validity bitmask: 1 bit / topk position, MASK_BYTES_PER_BUF = B_TOPK/8
    # bytes per buffer. Matches MLASparseSharedMemory so the reused
    # `kv_valid_producer` writes the identical layout WG0 consumes.
    comptime MASK_BYTES_PER_BUF = Self.config.B_TOPK // 8
    comptime NUM_KV_VALID_LANES = Self.MASK_BYTES_PER_BUF
    comptime INDICES_PER_LANE = 8

    # S-TMEM ring: QK(k) writes its score tile into a rotating slot instead
    # of one fixed address, so the MMA warp can race up to NUM_S_SLOTS
    # blocks ahead of the softmax consumer (previously hard-capped at 1
    # block regardless of num_mbars -- see mla_prefill_sparse_qkv_fp8.mojo's
    # `_mma`/`_softmax_epilogue` for the handoff this replaces, and
    # mla_decode_utils.mojo's `DecodeSProducerN`/`DecodeSConsumerN` for the
    # analogous decode-side ring this mirrors).  Slot stride follows the
    # SAME `MMA_N // 2` physical-column halving already used (unconditionally,
    # for both cg1 and cg2) by `O_ATOM_PHYS_COLS` in this file and in the
    # cg2-capable `mla_prefill_sparse_kv_fp8.mojo` sibling -- both proven
    # correct by their passing test suites -- and matches
    # `mla_decode_utils.mojo`'s own `TMEM_S1 - TMEM_S0 == 32` at BN_QK=64.
    # O occupies TMEM cols [0, 256) (2 atoms x O_ATOM_PHYS_COLS=128).  TMEM
    # fits up to (sm100_tmem_cols - S_TMEM_BASE) // S_SLOT_STRIDE = 8 ring
    # slots, but 4 is plenty: QK can already race 4 k-blocks ahead of softmax,
    # and these production shapes are KV-gather-latency bound (not S-handoff
    # bound), so a deeper ring buys no wall-clock while costing TMEM columns.
    # 4 slots use cols [256, 384); the upper half stays free for future use.
    comptime S_TMEM_BASE = 256
    comptime S_SLOT_STRIDE = Self.config.B_TOPK // 2
    comptime NUM_S_SLOTS = 4

    # FP8 operands. Q is persistent; KV/P are double-buffered for the QK(k) /
    # PV(k-1) one-block software-pipeline overlap. O (bf16) reuses the KV
    # region at the epilogue (all MMAs done -> KV free).
    var q: Array[Scalar[FP8_TYPE], Self.Q_SIZE]
    var kv: Array[Scalar[FP8_TYPE], Self.num_mbars * Self.KV_STAGE_SIZE]
    var p: Array[Scalar[FP8_TYPE], Self.num_mbars * Self.P_STAGE_SIZE]
    # Separate V (cg2).  Placed right after `p` so (a) q/kv/p keep their exact
    # HEAD byte offsets -- the SW64 MMA operands q/kv/p require 1024B
    # (swizzle-atom) base alignment, so they must NOT shift (a +384B shift of `p`
    # produced wrong numerics) -- and (b) `v` itself lands 1024B-aligned at cg2
    # (q/kv/p are all 1024B-multiple sized).  At cg1 it is a 128B-multiple dummy;
    # it only shifts the scalar/barrier arrays after it, which are field-pointer
    # addressed and alignment-insensitive.  O (bf16) reuses the `kv` region.
    var v: Array[Scalar[FP8_TYPE], Self.num_mbars * Self.V_STAGE_SIZE]

    # Physical gather rows for each in-flight KV block (K producer).
    var d_indices: Array[Int32, Self.num_mbars * Self.B_TOPK]
    # Separate V-producer gather rows (cg2: all B_TOPK keys; size 1 at cg1).
    var d_indices_v: Array[
        Int32, Self.num_mbars * (Self.B_TOPK if Self.is_cg2 else 1)
    ]

    var rowwise_max: Array[Float32, WARPGROUP_SIZE]
    var rowwise_sum: Array[Float32, WARPGROUP_SIZE]
    var is_k_valid: Array[UInt8, Self.num_mbars * Self.MASK_BYTES_PER_BUF]
    var tmem_addr: Array[UInt32, 1]

    var prologue_q: Array[SharedMemBarrier, 1]
    # QK^T MMA done (S slot in TMEM ready for WG0).  Sized to the S-TMEM
    # ring depth (NUM_S_SLOTS), NOT num_mbars -- decoupled from the KV SMEM
    # staging depth so the MMA warp can advance independently of it.
    var qk_done: Array[SharedMemBarrier, Self.NUM_S_SLOTS]
    # PV MMA done (O accumulated for the block).
    var sv_done: Array[SharedMemBarrier, Self.num_mbars]
    # K gather landed (local TMA transaction barrier).  At cg1 the MMA waits
    # this directly; at cg2 WG1 waits it, then arrives the cross-CTA k_ready.
    var kv_ready: Array[SharedMemBarrier, Self.num_mbars]
    # WG0 released an S-TMEM ring slot (see NUM_S_SLOTS above).
    var p_free: Array[SharedMemBarrier, Self.NUM_S_SLOTS]
    # WG0 wrote FP8 P + rescaled O (SV may proceed).
    var so_ready: Array[SharedMemBarrier, Self.num_mbars]
    var k_valid_ready: Array[SharedMemBarrier, Self.num_mbars]
    var k_valid_free: Array[SharedMemBarrier, Self.num_mbars]

    # cg2-only cluster handshake barriers (size 1 at cg1, unused there).  The
    # 2SM MMA on the leader (CTA0) reads BOTH CTAs' K/V/Q SMEM, so each CTA's
    # producer arrives the leader's cross-CTA barrier (init cta_group) after its
    # own local TMA lands.
    comptime CG2_MBARS = Self.num_mbars if Self.is_cg2 else 1
    var k_ready: Array[SharedMemBarrier, Self.CG2_MBARS]
    var v_ready: Array[SharedMemBarrier, Self.CG2_MBARS]
    var v_tma_done: Array[SharedMemBarrier, Self.CG2_MBARS]


struct MLAPrefillSparseQKVFP8[
    KVLUTType: MHAOperand,
    output_dtype: DType,
    config: MLASparseConfig,
](TrivialRegisterPassable):
    comptime accum_dtype = DType.float32
    # Linked FP8 P-scale / lazy-rescale knob, mirrored from
    # `mha_depth512/softmax_warp.mojo:396-413`.  The native path is always FP8
    # e4m3, so `RESCALE_THRESHOLD = -2` and `P_FP8_BIAS = 8 + RESCALE_THRESHOLD
    # = 6`.  P_FP8_BIAS lifts the un-normalized softmax probabilities P by
    # exp2(6)=64x out of the e4m3 subnormal floor before the FP8 cast that feeds
    # the P@V MMA (better PV quantization); it is an additive +bias in the exp2
    # argument (added raw, NOT * scale_log2e), so it scales P and the row-sum
    # `li` uniformly and cancels in O/li -- EXCEPT the attention-sink term, which
    # is biased too (below) to keep the cancellation.  The rescale gate fires
    # when a block's max lags the running max by more than |RESCALE_THRESHOLD|=2
    # (log2), reducing max-lag.  The two are LINKED for overflow safety: with
    # lag <= 2, max P = exp2(2 + 6) = exp2(8) = 256 < 448 (e4m3 max).
    comptime RESCALE_THRESHOLD: Float32 = -2.0
    comptime P_FP8_BIAS: Float32 = 8.0 + Self.RESCALE_THRESHOLD
    comptime qk_depth = Self.config.qk_depth
    comptime v_depth = Self.config.v_depth
    comptime cta_group = Self.config.cta_group
    comptime is_cg2 = Self.config.cta_group == 2
    # Real / padded head rows this CTA owns (head//cta_group at head=128).
    comptime NUM_Q_HEADS_PER_CTA = Self.config.num_q_heads // Self.config.cta_group
    comptime PADDED_HEADS = Self.config.padded_num_q_heads
    comptime PADDED_HEADS_PER_CTA = (
        Self.config.padded_num_q_heads // Self.config.cta_group
    )
    comptime B_TOPK = Self.config.B_TOPK
    comptime B_TOPK_PER_CTA = Self.config.B_TOPK // Self.config.cta_group
    # tcgen05.commit multicast mask: signal both CTAs in the pair at cg2.
    comptime CTA_MASK: UInt16 = 0b11 if Self.is_cg2 else 0b1

    comptime PV_BK = 64
    comptime NUM_PV_KHALVES = Self.B_TOPK // Self.PV_BK

    comptime NUM_SV_ATOMS = 2
    comptime SV_ATOM_MMA_N = Self.config.v_depth // Self.NUM_SV_ATOMS  # 256
    # Per-CTA V N-slice per atom (b_bmn); full 256 at cg1, 128 at cg2.
    comptime V_BMN_PER_ATOM = Self.SV_ATOM_MMA_N // Self.config.cta_group
    comptime V_SMEM_COLS_PER_CTA = Self.V_BMN_PER_ATOM * Self.NUM_SV_ATOMS
    comptime O_ATOM_PHYS_COLS = Self.SV_ATOM_MMA_N // 2  # 128

    comptime SMemType = MLASparseSharedMemoryQKVFP8[Self.config]

    # ---- TMA tile shapes ----
    comptime q_tile_shape = Index(
        1, Self.NUM_Q_HEADS_PER_CTA, Self.config.input_qk_depth
    )
    comptime q_desc_shape = _default_desc_shape[
        3, FP8_TYPE, Self.q_tile_shape, SW64
    ]()
    comptime Q_SWIZZLE_COLS = Self.q_desc_shape[2]

    comptime kv_gather_box = _gather4_box_width[
        FP8_TYPE, Self.config.input_qk_depth, SW64
    ]()
    comptime kv_tile_shape = Index(Self.B_TOPK, Self.kv_gather_box)
    comptime kv_desc_shape = Index(1, Self.kv_gather_box)

    # O TMA store: bf16 SWIZZLE_128B (unchanged from the BF16-KV kernel).
    comptime o_tile_shape = Index(Self.NUM_Q_HEADS_PER_CTA, Self.config.v_depth)
    comptime o_desc_shape = Index(Self.NUM_Q_HEADS_PER_CTA, 64)

    # ---- TMEM layout (512 cols) ----
    comptime O_TMEM_ADDR = 0
    comptime O_TMEM_ADDR_ATOM2 = Self.O_TMEM_ADDR + Self.O_ATOM_PHYS_COLS
    # S-TMEM ring geometry lives on SMemType (see the comment there) so the
    # SMEM barrier-array sizes and the TMEM address math share one source of
    # truth.
    comptime S_TMEM_BASE = Self.SMemType.S_TMEM_BASE
    comptime S_SLOT_STRIDE = Self.SMemType.S_SLOT_STRIDE
    comptime NUM_S_SLOTS = Self.SMemType.NUM_S_SLOTS

    # ---- Native-FP8 MMA accumulators (cta_group=1 warp-specialized ws) ----
    # QK^T: A=Q[64,576] k-major, B=K[64,576] k-major, transpose_b -> S[64,64].
    comptime QKAcc = SM100TensorAccumulator[
        FP8_TYPE,
        Self.accum_dtype,
        MMA_M=Self.PADDED_HEADS,
        MMA_N=Self.B_TOPK,
        BK=Self.config.qk_depth,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F8F6F4,
        swizzle_a=SW64,
        swizzle_b=SW64,
        transpose_b=True,
        cta_group=Self.config.cta_group,
    ]
    # PV per atom, per K-half: A=P[64,PV_BK] k-major, B=V[256,PV_BK] mn-major
    # -> O[64,256].  BK is always PV_BK (64, one SW64 atom), never B_TOPK
    # directly -- see NUM_PV_KHALVES above.
    comptime PVAcc = SM100TensorAccumulator[
        FP8_TYPE,
        Self.accum_dtype,
        MMA_M=Self.PADDED_HEADS,
        MMA_N=Self.SV_ATOM_MMA_N,
        BK=Self.PV_BK,
        a_tmem=False,
        mma_kind=UMMAKind.KIND_F8F6F4,
        swizzle_a=SW64,
        swizzle_b=SW64,
        transpose_b=False,
        cta_group=Self.config.cta_group,
    ]
    # Byte offset between the two V atoms WITHIN one K-half's region of the
    # (per-CTA) V buffer: atom `a` occupies V_BMN_PER_ATOM depth-cols x
    # PV_BK keys of the k-major SW64 tile the mn-major PV descriptor reads
    # (cg1: 256*64=16384; cg2: 128*64=8192).
    comptime V_ATOM_BYTE_OFFSET = (Self.V_BMN_PER_ATOM * Self.PV_BK) * size_of[
        FP8_TYPE
    ]()
    # Byte/element strides between consecutive K-halves in the P and V
    # buffers (each K-half is a self-contained tile of the SAME shape a
    # single-K-half kernel (B_TOPK==PV_BK, e.g. cg1 today) would produce,
    # stacked back to back -- see `_write_p_fp8` / `_load_v_fp8`).
    comptime P_KHALF_STRIDE = Self.PADDED_HEADS_PER_CTA * Self.PV_BK
    comptime V_KHALF_STRIDE = Self.V_SMEM_COLS_PER_CTA * Self.PV_BK

    comptime Common = MLAPrefillSparseCommon[
        Self.KVLUTType, Self.output_dtype, Self.config
    ]

    # ------------------------------------------------------------------
    # FP8 SW64 k-major P write (softmax exp2 numerators -> P SMEM).
    # Mirrors the decode `write_fp8_row_to_smem_chunked`
    # (make_ldmatrix_swizzle[fp8, row_size=PV_BK, lvw=4] address pattern that
    # the PV A descriptor `smem_descriptor[BK=PV_BK, SW64, k-major]` reads).
    # Writes 16 FP8 per STS.128 (4 uint32).  `query`/`key_base` come from WG0's
    # ws-packed S-TMEM read: query = idx%64, key-half = idx//64 (the "key-half"
    # thread-group split, which always spans P_PER_THREAD=B_TOPK/2 columns).
    # At cg1 (B_TOPK==PV_BK==64) `key_base` is < PV_BK for both groups, so
    # `half_idx` is always 0 and this reduces to the single-tile write
    # unchanged.  At cg2 (B_TOPK=2*PV_BK) the two thread-groups' key_base
    # values (0, PV_BK) land EXACTLY on the PV K-half boundary, so each group
    # writes one whole, independently-swizzled PV_BK-wide half-tile -- the
    # SAME physical layout `_mma`'s per-K-half BK=PV_BK descriptor reads.
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _write_p_fp8[
        n: Int
    ](
        p_smem: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        nums: Array[Float32, n],
        query: UInt32,
        key_base: UInt32,
    ):
        comptime P_SW = make_ldmatrix_swizzle[
            FP8_TYPE, row_size=Self.PV_BK, log2_vector_width=4
        ]()
        var half_idx = key_base / UInt32(Self.PV_BK)
        var local_key_base = key_base % UInt32(Self.PV_BK)
        var p_half_smem = p_smem + Int(half_idx) * Self.P_KHALF_STRIDE
        comptime for g in range(n // 16):
            comptime col_offset = g * 16
            var logical = Int(
                query * UInt32(Self.PV_BK) + local_key_base + UInt32(col_offset)
            )
            var phys = P_SW(logical)
            var packed = SIMD[.uint32, 4](0)
            comptime for sub in range(4):
                var f4 = SIMD[.float32, 4](
                    nums[g * 16 + sub * 4 + 0],
                    nums[g * 16 + sub * 4 + 1],
                    nums[g * 16 + sub * 4 + 2],
                    nums[g * 16 + sub * 4 + 3],
                )
                packed[sub] = bitcast[.uint32, 1](f4.cast[FP8_TYPE]())
            st_shared_v4_b32(p_half_smem, phys, packed)

    # ------------------------------------------------------------------
    # KV producer (WG1): compute physical gather rows, gather4 the FP8 SW64 KV
    # tile (K and V share it), signal kv_ready.  No dequant, no staging.
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _load_kv_fp8(
        kv_tma_op: TMATensorTile[
            FP8_TYPE, 2, Self.kv_tile_shape, Self.kv_desc_shape
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        kv_smem_buf: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        d_indices_buf: UnsafePointer[
            mut=True, Int32, address_space=.SHARED, ...
        ],
        kv_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k: UInt32,
        indices_base: UInt32,
        local_warp_idx: UInt32,
        cta_id: UInt32,
    ):
        # Map this CTA's B_TOPK_PER_CTA sparse indices for the block to physical
        # gather rows.  At cg2 each CTA owns the [cta_id*B_TOPK_PER_CTA, +...)
        # N-slice of K, written to local base 0 (the 2SM MMA sources each CTA's
        # K from its own local descriptor base).  Sentinels (raw < 0) stay
        # negative so the TMA zero-fills that row.
        var tid = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)
        if tid < UInt32(Self.B_TOPK_PER_CTA // 4):
            var indices_offset = (
                indices_base
                + k * UInt32(Self.config.B_TOPK)
                + cta_id * UInt32(Self.B_TOPK_PER_CTA)
                + tid * UInt32(4)
            )
            var rows = Self.Common._raw_indices_to_tma_rows(
                kv_lut,
                indices.load[width=4](Coord(indices_offset)).cast[
                    DType.int32
                ](),
            )
            comptime for j in range(4):
                d_indices_buf[Int(tid) * 4 + j] = rows[j]

        # Arm the transaction before flushing d_indices so the expect is
        # globally visible before any TMA arrival can satisfy the mbarrier.
        if local_warp_idx == 0 and elect_one_sync():
            kv_ready[].expect_bytes(
                Int32(Self.B_TOPK_PER_CTA * Self.config.input_qk_depth)
            )
        named_barrier[Int32(WARPGROUP_SIZE)](3)

        # Issue the gather4 col-group placement in parallel across WG1's 4
        # warps (the single-thread `async_copy_gather4_tile` serializes all
        # NUM_CG*NUM_CHUNKS issues). Each warp's elected lane owns a disjoint
        # 4-row-chunk range; every (chunk, col-group) lands at exactly the
        # SW64 offset `cg*BN*box_w + c*4*box_w` the k-major descriptor reads.
        comptime BN = Self.B_TOPK_PER_CTA
        comptime box_w = Self.kv_gather_box
        comptime NUM_CG = Self.config.input_qk_depth // box_w
        comptime NUM_CHUNKS = BN // 4
        comptime NUM_PROD_WARPS = WARPGROUP_SIZE // WARP_SIZE  # 4
        comptime CHUNKS_PER_WARP = NUM_CHUNKS // NUM_PROD_WARPS
        if elect_one_sync():
            comptime for lc in range(CHUNKS_PER_WARP):
                var c = local_warp_idx * UInt32(CHUNKS_PER_WARP) + UInt32(lc)
                var base_i = c * UInt32(4)
                var r0 = d_indices_buf[base_i]
                var r1 = d_indices_buf[base_i + 1]
                var r2 = d_indices_buf[base_i + 2]
                var r3 = d_indices_buf[base_i + 3]
                comptime for cg in range(NUM_CG):
                    var dst = TileTensor(
                        kv_smem_buf + Int(c) * 4 * box_w + cg * BN * box_w,
                        row_major[4, box_w](),
                    )
                    kv_tma_op.async_copy_gather4[cta_group=1](
                        dst, kv_ready[], Int32(cg * box_w), r0, r1, r2, r3
                    )

    # ------------------------------------------------------------------
    # V producer (WG2, cg2 only): gather ALL B_TOPK keys, this CTA's v_depth
    # column-slice, into the separate V buffer in the SAME k-major SW64 physical
    # layout the mn-major PV descriptor reads (so it transposes to V^T exactly
    # as the cg1 shared-buffer read does).  The native cg1 path never gathers V
    # (it shares the KV buffer); at cg2 K@base0 and full-V@base0 collide, so V
    # is separate.  Cross-ref dequant `load_v_fp8_tma` for the depth-split math.
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _load_v_fp8(
        kv_tma_op: TMATensorTile[
            FP8_TYPE, 2, Self.kv_tile_shape, Self.kv_desc_shape
        ],
        indices: TileTensor[.uint32, address_space=.GENERIC, ...],
        kv_lut: Self.KVLUTType,
        v_smem_buf: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        d_indices_buf: UnsafePointer[
            mut=True, Int32, address_space=.SHARED, ...
        ],
        v_tma_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k: UInt32,
        indices_base: UInt32,
        local_warp_idx: UInt32,
        cta_id: UInt32,
    ):
        # All B_TOPK key rows (V is NOT CTA-split by row; CTAs differ by which
        # depth columns they load, i.e. their MMA_N/cta_group N-slice).
        var tid = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)
        if tid < UInt32(Self.B_TOPK // 4):
            var indices_offset = (
                indices_base + k * UInt32(Self.config.B_TOPK) + tid * UInt32(4)
            )
            var rows = Self.Common._raw_indices_to_tma_rows(
                kv_lut,
                indices.load[width=4](Coord(indices_offset)).cast[
                    DType.int32
                ](),
            )
            comptime for j in range(4):
                d_indices_buf[Int(tid) * 4 + j] = rows[j]

        if local_warp_idx == 0 and elect_one_sync():
            v_tma_done[].expect_bytes(
                Int32(Self.B_TOPK * Self.V_SMEM_COLS_PER_CTA)
            )
        named_barrier[Int32(WARPGROUP_SIZE)](4)

        # Placement mirrors the K gather (k-major SW64: col-group `v_cg` at
        # `v_cg*PV_BK*box_w`, 4-row chunk at `local_c*4*box_w`), but the
        # B_TOPK topk rows are additionally split into NUM_PV_KHALVES
        # PV_BK-row halves, each stacked at `V_KHALF_STRIDE` -- the SAME
        # physical layout `_mma`'s per-K-half BK=PV_BK descriptor reads (see
        # the `PV_BK`/`NUM_PV_KHALVES` comment on the struct).  At cg1
        # (B_TOPK==PV_BK) NUM_PV_KHALVES==1 and this reduces to the
        # single-tile layout unchanged (khalf==0, local_c==c always).  The
        # source col-group for output col-group `v_cg` selects this CTA's
        # depth slice of atom `v_cg // CG_PER_ATOM_LOCAL`: global atom start
        # `atom*CG_PER_ATOM_GLOBAL` + this CTA's within-atom offset
        # `cta_id*CG_PER_ATOM_LOCAL` + `j`.
        comptime BN = Self.B_TOPK
        comptime box_w = Self.kv_gather_box
        comptime CG_PER_ATOM_GLOBAL = Self.SV_ATOM_MMA_N // box_w
        comptime CG_PER_ATOM_LOCAL = Self.V_BMN_PER_ATOM // box_w
        comptime NUM_V_CG = Self.V_SMEM_COLS_PER_CTA // box_w
        comptime NUM_CHUNKS = BN // 4
        comptime NUM_PROD_WARPS = WARPGROUP_SIZE // WARP_SIZE  # 4
        comptime CHUNKS_PER_WARP = NUM_CHUNKS // NUM_PROD_WARPS
        comptime CHUNKS_PER_KHALF = Self.PV_BK // 4
        if elect_one_sync():
            comptime for lc in range(CHUNKS_PER_WARP):
                var c = local_warp_idx * UInt32(CHUNKS_PER_WARP) + UInt32(lc)
                var khalf = c / UInt32(CHUNKS_PER_KHALF)
                var local_c = c % UInt32(CHUNKS_PER_KHALF)
                var base_i = c * UInt32(4)
                var r0 = d_indices_buf[base_i]
                var r1 = d_indices_buf[base_i + 1]
                var r2 = d_indices_buf[base_i + 2]
                var r3 = d_indices_buf[base_i + 3]
                comptime for v_cg in range(NUM_V_CG):
                    comptime atom = v_cg // CG_PER_ATOM_LOCAL
                    comptime j = v_cg % CG_PER_ATOM_LOCAL
                    var src_cg = (
                        UInt32(atom * CG_PER_ATOM_GLOBAL)
                        + cta_id * UInt32(CG_PER_ATOM_LOCAL)
                        + UInt32(j)
                    )
                    var dst = TileTensor(
                        v_smem_buf
                        + Int(khalf) * Self.V_KHALF_STRIDE
                        + Int(local_c) * 4 * box_w
                        + v_cg * Self.PV_BK * box_w,
                        row_major[4, box_w](),
                    )
                    kv_tma_op.async_copy_gather4[cta_group=1](
                        dst,
                        v_tma_done[],
                        Int32(src_cg * UInt32(box_w)),
                        r0,
                        r1,
                        r2,
                        r3,
                    )

    # ------------------------------------------------------------------
    # Q load prologue (FP8 SW64).  For head==64 a plain 3D async_copy; for
    # head<64 the padded 64-row buffer is zeroed and each real-head depth-tile
    # sub-copy lands at the PADDED stride the BMN=64 QK descriptor reads.
    # Mirrors `MLAPrefillSparseCommon._load_q_prologue` at SW64/FP8.
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _load_q_fp8(
        q_smem: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        q_tma_op: TMATensorTile[
            FP8_TYPE, 3, Self.q_tile_shape, Self.q_desc_shape
        ],
        prologue_q: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        seq_idx: UInt32,
        cta_id: UInt32,
    ):
        # Each CTA loads only its NUM_Q_HEADS_PER_CTA real heads (at cg2, heads
        # [cta_id*NUM_Q_HEADS_PER_CTA, ...)).  The TMA head coord is in ELEMENTS.
        # At cg2 the leader arms prologue_q with cta_group*bytes and waits it;
        # here (cg1) warp0 arms its own single-CTA byte count.
        comptime if Self.NUM_Q_HEADS_PER_CTA < Self.PADDED_HEADS_PER_CTA:
            for i in range(
                Int(thread_idx.x),
                Self.SMemType.Q_SIZE,
                Self.config.num_threads,
            ):
                q_smem[i] = Scalar[FP8_TYPE](0)
            barrier()
            fence_async_view_proxy()

        if warp_id() == 0 and elect_one_sync():
            comptime q_bytes = (
                Self.NUM_Q_HEADS_PER_CTA * Self.config.input_qk_depth
            )
            comptime if not Self.is_cg2:
                prologue_q[].expect_bytes(Int32(q_bytes))
            var head_coord = cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA)
            comptime if Self.NUM_Q_HEADS_PER_CTA < Self.PADDED_HEADS_PER_CTA:
                comptime NUM_Q_DEPTH_TILES = (
                    Self.config.input_qk_depth // Self.Q_SWIZZLE_COLS
                )
                comptime Q_PADDED_DEPTH_TILE_STRIDE = (
                    Self.PADDED_HEADS_PER_CTA * Self.Q_SWIZZLE_COLS
                )
                comptime for j in range(NUM_Q_DEPTH_TILES):
                    cp_async_bulk_tensor_shared_cluster_global[
                        cta_group=Self.config.cta_group
                    ](
                        q_smem + j * Q_PADDED_DEPTH_TILE_STRIDE,
                        UnsafePointer(to=q_tma_op.descriptor).bitcast[
                            NoneType
                        ](),
                        prologue_q[].unsafe_ptr(),
                        Index(
                            j * Self.Q_SWIZZLE_COLS,
                            Int(head_coord),
                            Int(seq_idx),
                        ),
                    )
            else:
                var q_full = TileTensor(
                    q_smem,
                    row_major[
                        1,
                        Self.NUM_Q_HEADS_PER_CTA,
                        Self.config.input_qk_depth,
                    ](),
                )
                q_tma_op.async_copy[cta_group=Self.config.cta_group](
                    q_full,
                    prologue_q[],
                    StaticTuple[UInt32, 3](0, head_coord, seq_idx),
                )

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(kv_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma_op, `nvvm.grid_constant`)
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](
            Int32(Self.config.cta_group), 1, 1
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__name(
        t"mla_prefill_sparse_qkv_fp8_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}_iqd{Self.config.input_qk_depth}",
    )
    def kernel[
        TopKLengthLayout: TensorLayout,
        IndicesLayout: TensorLayout,
        TopKLengthEngine: TensorEngine,
        IndicesEngine: TensorEngine,
    ](
        q_tma_op: TMATensorTile[
            FP8_TYPE, 3, Self.q_tile_shape, Self.q_desc_shape
        ],
        kv_tma_op: TMATensorTile[
            FP8_TYPE, 2, Self.kv_tile_shape, Self.kv_desc_shape
        ],
        o_tma_op: TMATensorTile[
            Self.output_dtype, 2, Self.o_tile_shape, Self.o_desc_shape
        ],
        topk_lengths: TileTensor[
            .uint32, TopKLengthLayout, MutAnyOrigin, Engine=TopKLengthEngine
        ],
        indices: TileTensor[
            .uint32, IndicesLayout, MutAnyOrigin, Engine=IndicesEngine
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
        indices_stride: Int32,
    ) where (topk_lengths.flat_rank == 1 and indices.flat_rank == 1):
        var cta_id = UInt32(block_idx.x % Self.config.cta_group)
        var seq_idx = UInt32(block_idx.x // Self.config.cta_group)
        var warp_idx = warp_id()
        var lane_idx = thread_idx.x % WARP_SIZE
        var warpgroup_idx = warp.broadcast(thread_idx.x // WARPGROUP_SIZE)
        var top_k_length = topk_lengths.load[width=1](Coord(seq_idx))
        var num_k_blocks = max(
            ceildiv(top_k_length, UInt32(Self.config.B_TOPK)), 1
        )
        var num_kv_rows = kv_lut.num_kv_rows()
        var indices_base = seq_idx * UInt32(indices_stride)

        if thread_idx.x == 0:
            q_tma_op.prefetch_descriptor()
            kv_tma_op.prefetch_descriptor()

        ref smem = external_memory[
            UInt8, address_space=.SHARED, alignment=128
        ]().bitcast[Self.SMemType]()[]

        var q_ptr = smem.q.unsafe_ptr()
        var kv_ptr: UnsafePointer[
            Scalar[FP8_TYPE], origin_of(smem.kv), address_space=.SHARED
        ] = smem.kv.unsafe_ptr()
        var v_ptr: UnsafePointer[
            Scalar[FP8_TYPE], origin_of(smem.v), address_space=.SHARED
        ] = smem.v.unsafe_ptr()
        var p_ptr: UnsafePointer[
            Scalar[FP8_TYPE], origin_of(smem.p), address_space=.SHARED
        ] = smem.p.unsafe_ptr()
        var d_indices_ptr: UnsafePointer[
            Int32, origin_of(smem.d_indices), address_space=.SHARED
        ] = smem.d_indices.unsafe_ptr()
        var d_indices_v_ptr: UnsafePointer[
            Int32, origin_of(smem.d_indices_v), address_space=.SHARED
        ] = smem.d_indices_v.unsafe_ptr()
        var rowwise_max_ptr: UnsafePointer[
            Float32, origin_of(smem.rowwise_max), address_space=.SHARED
        ] = smem.rowwise_max.unsafe_ptr()
        var rowwise_sum_ptr: UnsafePointer[
            Float32, origin_of(smem.rowwise_sum), address_space=.SHARED
        ] = smem.rowwise_sum.unsafe_ptr()
        var is_k_valid_ptr: UnsafePointer[
            UInt8, origin_of(smem.is_k_valid), address_space=.SHARED
        ] = smem.is_k_valid.unsafe_ptr()
        var tmem_addr_ptr = smem.tmem_addr.unsafe_ptr()
        var prologue_q_ptr = smem.prologue_q.unsafe_ptr()
        var qk_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.qk_done), address_space=.SHARED
        ] = smem.qk_done.unsafe_ptr()
        var sv_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.sv_done), address_space=.SHARED
        ] = smem.sv_done.unsafe_ptr()
        var kv_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.kv_ready), address_space=.SHARED
        ] = smem.kv_ready.unsafe_ptr()
        var p_free_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.p_free), address_space=.SHARED
        ] = smem.p_free.unsafe_ptr()
        var so_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.so_ready), address_space=.SHARED
        ] = smem.so_ready.unsafe_ptr()
        var k_valid_ready_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem.k_valid_ready),
            address_space=.SHARED,
        ] = smem.k_valid_ready.unsafe_ptr()
        var k_valid_free_ptr: UnsafePointer[
            SharedMemBarrier,
            origin_of(smem.k_valid_free),
            address_space=.SHARED,
        ] = smem.k_valid_free.unsafe_ptr()
        var k_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.k_ready), address_space=.SHARED
        ] = smem.k_ready.unsafe_ptr()
        var v_ready_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.v_ready), address_space=.SHARED
        ] = smem.v_ready.unsafe_ptr()
        var v_tma_done_ptr: UnsafePointer[
            SharedMemBarrier, origin_of(smem.v_tma_done), address_space=.SHARED
        ] = smem.v_tma_done.unsafe_ptr()

        if warp_idx == 0:
            if elect_one_sync():
                prologue_q_ptr[].init(1)
                comptime for i in range(Self.config.num_mbars):
                    sv_done_ptr[i].init(1)
                    # Local K TMA transaction barrier.
                    kv_ready_ptr[i].init(1)
                    so_ready_ptr[i].init(
                        Int32(WARPGROUP_SIZE * Self.config.cta_group)
                    )
                    k_valid_ready_ptr[i].init(
                        Int32(Self.SMemType.NUM_KV_VALID_LANES)
                    )
                    k_valid_free_ptr[i].init(Int32(WARPGROUP_SIZE))
                # S-TMEM ring barriers: sized/indexed by NUM_S_SLOTS, not
                # num_mbars (see the SMemType field comments).  p_free is
                # consumed by the MMA leader from BOTH CTAs' WG0 at cg2 ->
                # init WARPGROUP_SIZE*cta_group and arrive_cluster;
                # single-CTA count at cg1.
                comptime for i in range(Self.SMemType.NUM_S_SLOTS):
                    qk_done_ptr[i].init(1)
                    p_free_ptr[i].init(
                        Int32(WARPGROUP_SIZE * Self.config.cta_group)
                    )
                comptime if Self.is_cg2:
                    comptime for i in range(Self.SMemType.CG2_MBARS):
                        # Cross-CTA: each CTA's producer arrives once (the 2SM
                        # MMA reads both CTAs' K/V), so init count = cta_group.
                        k_ready_ptr[i].init(Int32(Self.config.cta_group))
                        v_ready_ptr[i].init(Int32(Self.config.cta_group))
                        # Local V TMA transaction barrier.
                        v_tma_done_ptr[i].init(1)
                fence_mbarrier_init()

        # Zero the NoPE tail (Q and each KV buffer). `p` is its own buffer and V
        # occupies only the leading v_depth columns, so nothing rewrites the tail
        # afterwards. The fence publishes the stores to the async proxy for the
        # tcgen05 read.
        comptime if Self.config.input_qk_depth < Self.config.qk_depth:
            comptime _q_kept = (
                Self.SMemType.PADDED_HEADS_PER_CTA * Self.config.input_qk_depth
            )
            comptime _kv_kept = (
                Self.SMemType.B_TOPK_PER_CTA * Self.config.input_qk_depth
            )
            var _tid = Int(thread_idx.x)
            for i in range(
                _q_kept + _tid,
                Self.SMemType.Q_SIZE,
                Self.config.num_threads,
            ):
                q_ptr[i] = Scalar[FP8_TYPE](0)
            comptime for _st in range(Self.SMemType.num_mbars):
                var _base = _st * Self.SMemType.KV_STAGE_SIZE
                for i in range(
                    _kv_kept + _tid,
                    Self.SMemType.KV_STAGE_SIZE,
                    Self.config.num_threads,
                ):
                    kv_ptr[_base + i] = Scalar[FP8_TYPE](0)
            fence_async_view_proxy()
        cluster_sync()

        Self._load_q_fp8(q_ptr, q_tma_op, prologue_q_ptr, seq_idx, cta_id)

        if warp_idx == 0:
            tcgen05_alloc[Int32(Self.config.cta_group)](
                tmem_addr_ptr, Self.config.sm100_tmem_cols
            )
            tcgen05_release_allocation_lock[Int32(Self.config.cta_group)]()
        barrier()

        if warpgroup_idx == 0:
            comptime if Self.is_cg2:
                warpgroup_reg_alloc[240]()
            else:
                warpgroup_reg_alloc[168]()
            Self._softmax_epilogue(
                p_ptr,
                kv_ptr,
                rowwise_max_ptr,
                rowwise_sum_ptr,
                is_k_valid_ptr,
                sv_done_ptr,
                qk_done_ptr,
                p_free_ptr,
                so_ready_ptr,
                k_valid_ready_ptr,
                k_valid_free_ptr,
                tmem_addr_ptr,
                o_tma_op,
                scale,
                attn_sink_ptr,
                num_k_blocks,
                seq_idx,
                cta_id,
            )
        elif warpgroup_idx == 1:
            # K producer.  cg2 frees WG2 to be the V producer; both cg2
            # producers dealloc to 64 (fp8 gather is light -- no dequant
            # staging) so WG0's softmax epilogue can absorb the freed regs,
            # mirroring cg1's WG2/WG3 dealloc[88] idle/mma pattern.
            comptime if Self.is_cg2:
                warpgroup_reg_dealloc[64]()
            else:
                warpgroup_reg_alloc[168]()
            var local_warp_idx = UInt32(warp_id() - 4)
            for k in range(num_k_blocks):
                var buf = k % UInt32(Self.config.num_mbars)
                var phase = (k / UInt32(Self.config.num_mbars)) & 1
                # Free the previous occupant (block k-num_mbars): cg1's shared
                # buffer frees after PV (sv_done); cg2's K-only buffer frees
                # after QK consumes it (qk_done).  qk_done is indexed by
                # NUM_S_SLOTS (the S-ring modulus), NOT num_mbars -- the K
                # SMEM buffer-free gating condition (k >= num_mbars) and the
                # specific block being freed (free_k) are unchanged, only the
                # array we query for "has QK(free_k) fired" uses its own
                # (decoupled) modulus.
                if k >= UInt32(Self.config.num_mbars):
                    var free_k = k - UInt32(Self.config.num_mbars)
                    comptime if Self.is_cg2:
                        var free_buf_s = free_k % UInt32(
                            Self.SMemType.NUM_S_SLOTS
                        )
                        var free_phase_s = (
                            free_k / UInt32(Self.SMemType.NUM_S_SLOTS)
                        ) & 1
                        qk_done_ptr[free_buf_s].wait(free_phase_s)
                    else:
                        var free_buf = free_k % UInt32(Self.config.num_mbars)
                        var free_phase = (
                            free_k / UInt32(Self.config.num_mbars)
                        ) & 1
                        sv_done_ptr[free_buf].wait(free_phase)
                Self._load_kv_fp8(
                    kv_tma_op,
                    indices,
                    kv_lut,
                    kv_ptr + buf * UInt32(Self.SMemType.KV_STAGE_SIZE),
                    d_indices_ptr + buf * UInt32(Self.config.B_TOPK),
                    kv_ready_ptr + buf,
                    k,
                    indices_base,
                    local_warp_idx,
                    cta_id,
                )
                # cg2: wait the local K TMA, then arrive the leader's cross-CTA
                # k_ready (the 2SM MMA on CTA0 reads BOTH CTAs' K).
                comptime if Self.is_cg2:
                    kv_ready_ptr[buf].wait(phase)
                    if local_warp_idx == 0 and elect_one_sync():
                        k_ready_ptr[buf].arrive_cluster(UInt32(0))
        elif warpgroup_idx == 2:
            comptime if Self.is_cg2:
                # V producer (cg2): gather all B_TOPK keys' per-CTA depth
                # slice.  Dealloc to 64, same rationale as WG1 above -- this
                # WG previously had no setmaxnreg call at all and fell
                # through to the 128-reg launch default.
                warpgroup_reg_dealloc[64]()
                var local_warp_idx = UInt32(warp_id() - 8)
                for k in range(num_k_blocks):
                    var buf = k % UInt32(Self.config.num_mbars)
                    var phase = (k / UInt32(Self.config.num_mbars)) & 1
                    # Free the V buffer's previous occupant after PV consumed it.
                    if k >= UInt32(Self.config.num_mbars):
                        var free_k = k - UInt32(Self.config.num_mbars)
                        var free_buf = free_k % UInt32(Self.config.num_mbars)
                        var free_phase = (
                            free_k / UInt32(Self.config.num_mbars)
                        ) & 1
                        sv_done_ptr[free_buf].wait(free_phase)
                    Self._load_v_fp8(
                        kv_tma_op,
                        indices,
                        kv_lut,
                        v_ptr + buf * UInt32(Self.SMemType.V_STAGE_SIZE),
                        d_indices_v_ptr + buf * UInt32(Self.config.B_TOPK),
                        v_tma_done_ptr + buf,
                        k,
                        indices_base,
                        local_warp_idx,
                        cta_id,
                    )
                    v_tma_done_ptr[buf].wait(phase)
                    if local_warp_idx == 0 and elect_one_sync():
                        v_ready_ptr[buf].arrive_cluster(UInt32(0))
            else:
                # V shares the KV buffer -> no separate V producer. Idle.
                warpgroup_reg_dealloc[88]()
        else:
            warpgroup_reg_dealloc[88]()
            # Only the leader CTA issues the 2SM MMA; it drives both CTAs' TMEM.
            if cta_id == 0 and warp_idx == 12 and elect_one_sync():
                Self._mma(
                    q_ptr,
                    kv_ptr,
                    p_ptr,
                    prologue_q_ptr,
                    kv_ready_ptr,
                    v_ptr,
                    k_ready_ptr,
                    v_ready_ptr,
                    qk_done_ptr,
                    sv_done_ptr,
                    so_ready_ptr,
                    p_free_ptr,
                    num_k_blocks,
                )
            elif warp_idx == 13 and lane_idx < Self.SMemType.NUM_KV_VALID_LANES:
                Self.Common.kv_valid_producer(
                    indices,
                    is_k_valid_ptr,
                    k_valid_ready_ptr,
                    k_valid_free_ptr,
                    UInt32(lane_idx),
                    indices_base,
                    Int32(num_kv_rows),
                    Int32(top_k_length),
                    Int(num_k_blocks),
                )

    # ------------------------------------------------------------------
    # MMA warp (WG3 warp 12, single elected lane): QK^T(k) then PV(k-1).
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _mma(
        q_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        kv_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        p_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        prologue_q: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        kv_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        v_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        k_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        v_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        qk_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        sv_done: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        so_ready: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        p_free: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        num_k_blocks: UInt32,
    ):
        var e = elect()
        # At cg2 the 2SM QK reads BOTH CTAs' Q SMEM; each CTA's local Q TMA
        # routes its completion to the leader's prologue_q, so the leader
        # expects cta_group * per-CTA bytes.  cg1 armed expect in `_load_q_fp8`.
        comptime if Self.is_cg2:
            prologue_q[].expect_bytes(
                Int32(
                    Self.NUM_Q_HEADS_PER_CTA
                    * Self.config.input_qk_depth
                    * Self.config.cta_group
                )
            )
        prologue_q[].wait()
        tcgen05_fence_after()

        for k in range(num_k_blocks + 1):
            if k < num_k_blocks:
                var buf = k % UInt32(Self.config.num_mbars)
                var phase = (k / UInt32(Self.config.num_mbars)) & 1
                # cg1: MMA waits the local K TMA barrier directly.  cg2: waits
                # the cross-CTA k_ready that both CTAs arrive after their local
                # K gather (the 2SM MMA reads both CTAs' K).
                comptime if Self.is_cg2:
                    k_ready[buf].wait(phase)
                else:
                    kv_ready[buf].wait(phase)
                # S-TMEM ring: QK(k) writes into slot (k % NUM_S_SLOTS).  The
                # slot is free once softmax has consumed the block that last
                # occupied it (k - NUM_S_SLOTS), NOT just the immediately
                # preceding block -- this is what lets QK race up to
                # NUM_S_SLOTS ahead of softmax instead of exactly 1.
                var s_slot = k % UInt32(Self.NUM_S_SLOTS)
                if k >= UInt32(Self.NUM_S_SLOTS):
                    var free_k_s = k - UInt32(Self.NUM_S_SLOTS)
                    var free_phase_s = (free_k_s / UInt32(Self.NUM_S_SLOTS)) & 1
                    p_free[s_slot].wait(free_phase_s)
                tcgen05_fence_after()
                var q_desc = smem_descriptor[
                    BMN=Self.PADDED_HEADS_PER_CTA,
                    BK=Self.config.qk_depth,
                    swizzle_mode=SW64,
                    is_k_major=True,
                ](q_ptr)
                var k_desc = smem_descriptor[
                    BMN=Self.B_TOPK_PER_CTA,
                    BK=Self.config.qk_depth,
                    swizzle_mode=SW64,
                    is_k_major=True,
                ](kv_ptr + buf * UInt32(Self.SMemType.KV_STAGE_SIZE))
                var s_tmem_addr = UInt32(Self.S_TMEM_BASE) + s_slot * UInt32(
                    Self.S_SLOT_STRIDE
                )
                Self.QKAcc.mma(
                    q_desc,
                    k_desc,
                    s_tmem_addr,
                    c_scale=0,
                    elect=e,
                )
                mma_arrive_multicast[cta_group=Self.config.cta_group](
                    qk_done[s_slot].unsafe_ptr(), Self.CTA_MASK
                )

            if k > 0:
                var pbuf = (k - 1) % UInt32(Self.config.num_mbars)
                var pphase = ((k - 1) / UInt32(Self.config.num_mbars)) & 1
                so_ready[pbuf].wait(pphase)
                # cg2: wait the cross-CTA v_ready (both CTAs' V gathered).  cg1:
                # V shares the K buffer already gated by kv_ready above.
                comptime if Self.is_cg2:
                    v_ready[pbuf].wait(pphase)
                tcgen05_fence_after()
                # cg1 reads V mn-major from the shared K buffer; cg2 from the
                # separate per-CTA V buffer.
                var v_base = (
                    v_ptr + pbuf * UInt32(Self.SMemType.V_STAGE_SIZE)
                ) if Self.is_cg2 else (
                    kv_ptr + pbuf * UInt32(Self.SMemType.KV_STAGE_SIZE)
                )
                var p_base = p_ptr + pbuf * UInt32(Self.SMemType.P_STAGE_SIZE)
                var c_scale_sv: UInt32 = 0 if k == 1 else 1
                # K-axis split: NUM_PV_KHALVES independent BK=PV_BK sub-MMAs
                # per atom, each reading its own self-contained P/V half-tile
                # (see `_write_p_fp8`/`_load_v_fp8`) and accumulating into the
                # SAME O TMEM address.  At cg1 NUM_PV_KHALVES==1, so this loop
                # is a single comptime-unrolled iteration -- byte-identical
                # codegen to the pre-split single-BK=B_TOPK MMA.
                comptime for khalf in range(Self.NUM_PV_KHALVES):
                    var p_desc = smem_descriptor[
                        BMN=Self.PADDED_HEADS_PER_CTA,
                        BK=Self.PV_BK,
                        swizzle_mode=SW64,
                        is_k_major=True,
                    ](p_base + UInt32(khalf * Self.P_KHALF_STRIDE))
                    var v_desc = smem_descriptor[
                        BMN=Self.V_BMN_PER_ATOM,
                        BK=Self.PV_BK,
                        swizzle_mode=SW64,
                        is_k_major=False,
                    ](v_base + UInt32(khalf * Self.V_KHALF_STRIDE))
                    # khalf 0 obeys the block's accum_init (c_scale_sv);
                    # khalf 1+ always accumulate onto khalf 0's partial sum.
                    var c_scale: UInt32 = c_scale_sv if khalf == 0 else 1
                    Self.PVAcc.mma(
                        p_desc,
                        v_desc,
                        UInt32(Self.O_TMEM_ADDR),
                        c_scale=c_scale,
                        elect=e,
                    )
                    Self.PVAcc.mma(
                        p_desc,
                        v_desc + UInt32(Self.V_ATOM_BYTE_OFFSET),
                        UInt32(Self.O_TMEM_ADDR_ATOM2),
                        c_scale=c_scale,
                        elect=e,
                    )
                mma_arrive_multicast[cta_group=Self.config.cta_group](
                    sv_done[pbuf].unsafe_ptr(), Self.CTA_MASK
                )

    # ------------------------------------------------------------------
    # Softmax + FP8 P write + O rescale + epilogue (WG0, warps 0-3).
    # ------------------------------------------------------------------
    @always_inline
    @staticmethod
    def _softmax_epilogue(
        p_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        kv_ptr: UnsafePointer[
            mut=True, Scalar[FP8_TYPE], address_space=.SHARED, ...
        ],
        rowwise_max_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
        rowwise_sum_ptr: UnsafePointer[
            mut=True, Float32, address_space=.SHARED, ...
        ],
        is_k_valid_ptr: UnsafePointer[
            mut=True, UInt8, address_space=.SHARED, ...
        ],
        sv_done_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        qk_done_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        p_free_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        so_ready_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k_valid_ready_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        k_valid_free_ptr: UnsafePointer[
            mut=True, SharedMemBarrier, address_space=.SHARED, ...
        ],
        tmem_addr_ptr: UnsafePointer[
            mut=True, UInt32, address_space=.SHARED, ...
        ],
        o_tma_op: TMATensorTile[
            Self.output_dtype, 2, Self.o_tile_shape, Self.o_desc_shape
        ],
        scale: Float32,
        attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
        num_k_blocks: UInt32,
        seq_idx: UInt32,
        cta_id: UInt32,
    ):
        var warp_idx = warp_id()
        var lane_idx = thread_idx.x % WARP_SIZE
        var idx_in_wg = UInt32(thread_idx.x) % UInt32(WARPGROUP_SIZE)

        comptime MAX_INIT_VAL = Float32(-1e30)
        var mi: Float32 = MAX_INIT_VAL
        var li: Float32 = 0.0
        var real_mi: Float32 = Float32(min_or_neg_inf[.float32]())

        var scale_log2e = scale * Float32(log2e)
        comptime P_PER_THREAD = Self.B_TOPK // 2
        comptime O_RESCALE_CHUNK = 32
        comptime NUM_O_RESCALE_CHUNKS = Self.SV_ATOM_MMA_N // O_RESCALE_CHUNK

        var query = idx_in_wg % UInt32(64)
        var key_base = (idx_in_wg / UInt32(64)) * UInt32(P_PER_THREAD)

        for k in range(num_k_blocks):
            var cur_buf = k % UInt32(Self.config.num_mbars)
            var cur_phase = (k / UInt32(Self.config.num_mbars)) & 1
            # S-TMEM ring slot for this block (decoupled from num_mbars --
            # see the SMemType field comments and `_mma` above).
            var s_slot = k % UInt32(Self.NUM_S_SLOTS)
            var s_phase = (k / UInt32(Self.NUM_S_SLOTS)) & 1

            qk_done_ptr[s_slot].wait(s_phase)
            tcgen05_fence_after()

            var p = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=P_PER_THREAD,
                dtype=DType.float32,
                pack=False,
                width=P_PER_THREAD,
            ](UInt32(Self.S_TMEM_BASE) + s_slot * UInt32(Self.S_SLOT_STRIDE))
            tcgen05_load_wait()
            tcgen05_fence_before()
            # cg2: the MMA leader (CTA0) reuses this S-TMEM slot once BOTH
            # CTAs' WG0 have read it, so arrive the leader's cross-CTA p_free.
            comptime if Self.is_cg2:
                p_free_ptr[s_slot].arrive_cluster(UInt32(0), UInt32(1))
            else:
                _ = p_free_ptr[s_slot].arrive()

            comptime MASK_BYTES_PER_BUF = Self.SMemType.MASK_BYTES_PER_BUF
            comptime MASK_BYTES_PER_THREAD = MASK_BYTES_PER_BUF // 2
            k_valid_ready_ptr[cur_buf].wait(cur_phase)
            var mask_byte_base = (
                Int(cur_buf) * MASK_BYTES_PER_BUF
                + Int(idx_in_wg // UInt32(64)) * MASK_BYTES_PER_THREAD
            )
            comptime for i in range(P_PER_THREAD):
                comptime byte_offset = i // 8
                comptime bit_idx = i % 8
                var mask_byte = is_k_valid_ptr[mask_byte_base + byte_offset]
                if ((mask_byte >> UInt8(bit_idx)) & UInt8(1)) == UInt8(0):
                    p[i] = Float32(min_or_neg_inf[.float32]())
            _ = k_valid_free_ptr[cur_buf].arrive()

            var cur_pi_max: Float32 = Float32(min_or_neg_inf[.float32]())
            comptime for i in range(P_PER_THREAD):
                cur_pi_max = max(cur_pi_max, p[i])
            cur_pi_max = mul_ftz(cur_pi_max, scale_log2e)

            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            rowwise_max_ptr[idx_in_wg] = cur_pi_max
            named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
            cur_pi_max = max(
                cur_pi_max, rowwise_max_ptr[idx_in_wg ^ UInt32(64)]
            )
            real_mi = max(real_mi, cur_pi_max)

            # Per-lane decision for the STATE update (new_max/mi/li):
            # `idx_in_wg` packs one independent head-row's softmax state
            # per lane, so an OR here would leak a sibling head's rescale
            # trajectory into this one's.
            var should_scale_o = cur_pi_max - mi > Float32(
                -Self.RESCALE_THRESHOLD
            )

            var new_max: Float32
            var scale_for_old: Float32
            if not should_scale_o:
                scale_for_old = 1.0
                new_max = mi
            else:
                new_max = max(cur_pi_max, mi)
                scale_for_old = exp2(mi - new_max)
            mi = new_max
            li = mul_ftz(li, scale_for_old)

            # The O-rescale WALK below touches TMEM via tcgen05_ld/st,
            # which are warp-collective ops (datapaths=32) requiring every
            # lane to participate uniformly -- a per-lane branch here would
            # diverge the warp on those ops and hang. Vote ANY (not the
            # state above): any lane needing a rescale pulls every lane
            # through the walk, but each lane applies its OWN
            # `scale_for_old` (exactly 1.0 for a lane that didn't need
            # it), so a coerced lane's contribution is an exact no-op
            # multiply, not a value substitution.
            var any_rescale = warp.vote[.uint32](should_scale_o) != UInt32(0)

            var nums = Array[Float32, P_PER_THREAD](uninitialized=True)
            # +P_FP8_BIAS lifts P out of the e4m3 subnormal floor; it scales
            # `li` and every P uniformly and cancels in O/li (and in the biased
            # sink term below).  Overflow-safe: max P = exp2(|RESCALE_THRESHOLD|
            # + P_FP8_BIAS) = exp2(8) = 256 < 448 (e4m3 max).
            comptime assert (
                -Self.RESCALE_THRESHOLD + Self.P_FP8_BIAS <= 8.0
            ), "FP8 P bias would overflow e4m3 (max P must be <= exp2(8)=256)"
            comptime for i in range(P_PER_THREAD):
                var d: Float32 = p[i] * scale_log2e - new_max + Self.P_FP8_BIAS
                var ed: Float32 = exp2(d)
                li = li + ed
                nums[i] = ed

            if k > 0:
                var prev_buf = (k - 1) % UInt32(Self.config.num_mbars)
                var prev_phase = ((k - 1) / UInt32(Self.config.num_mbars)) & 1
                sv_done_ptr[prev_buf].wait(prev_phase)

            var o_chunk_prefetch = Array[Float32, O_RESCALE_CHUNK](
                uninitialized=True
            )
            if k > 0 and any_rescale:
                tcgen05_fence_after()
                o_chunk_prefetch = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=O_RESCALE_CHUNK,
                    dtype=DType.float32,
                    pack=False,
                    width=O_RESCALE_CHUNK,
                ](UInt32(Self.O_TMEM_ADDR))

            # Write FP8 P (exp2 numerators) into the double-buffered P SMEM in
            # the SW64 k-major layout the PV A descriptor reads.
            Self._write_p_fp8[P_PER_THREAD](
                p_ptr + cur_buf * UInt32(Self.SMemType.P_STAGE_SIZE),
                nums,
                query,
                key_base,
            )

            if k > 0 and any_rescale:
                tcgen05_load_wait()
                var o_scaled_0 = Array[_, O_RESCALE_CHUNK](
                    fill_with_unrolled=lambda [j: Int]() -> Float32: (
                        mul_ftz(o_chunk_prefetch[j], scale_for_old)
                    )
                )
                tcgen05_st[
                    datapaths=32, bits=32, repeat=O_RESCALE_CHUNK, pack=False
                ](UInt32(Self.O_TMEM_ADDR), o_scaled_0)
                comptime for chunk_idx in range(1, NUM_O_RESCALE_CHUNKS):
                    var o_chunk = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=O_RESCALE_CHUNK,
                        dtype=DType.float32,
                        pack=False,
                        width=O_RESCALE_CHUNK,
                    ](
                        UInt32(Self.O_TMEM_ADDR)
                        + UInt32(chunk_idx * O_RESCALE_CHUNK)
                    )
                    tcgen05_load_wait()
                    var o_scaled = Array[_, O_RESCALE_CHUNK](
                        fill_with_unrolled=lambda [j: Int]() -> Float32: (
                            mul_ftz(o_chunk[j], scale_for_old)
                        )
                    )
                    tcgen05_st[
                        datapaths=32,
                        bits=32,
                        repeat=O_RESCALE_CHUNK,
                        pack=False,
                    ](
                        UInt32(Self.O_TMEM_ADDR)
                        + UInt32(chunk_idx * O_RESCALE_CHUNK),
                        o_scaled,
                    )
                tcgen05_store_wait()
                tcgen05_fence_before()

            fence_async_view_proxy()
            comptime if Self.is_cg2:
                so_ready_ptr[cur_buf].arrive_cluster(UInt32(0), UInt32(1))
            else:
                _ = so_ready_ptr[cur_buf].arrive()

        if real_mi == Float32(min_or_neg_inf[.float32]()):
            li = 0.0
            mi = Float32(min_or_neg_inf[.float32]())

        rowwise_sum_ptr[idx_in_wg] = li
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
        li = add_ftz(li, rowwise_sum_ptr[idx_in_wg ^ UInt32(64)])

        var last_buf = (num_k_blocks - 1) % UInt32(Self.config.num_mbars)
        var last_phase = (
            (num_k_blocks - 1) / UInt32(Self.config.num_mbars)
        ) & 1
        sv_done_ptr[last_buf].wait(last_phase)
        tcgen05_fence_after()

        var output_scale: Float32
        var sink_row = idx_in_wg % UInt32(64)
        if attn_sink_ptr and sink_row < UInt32(Self.NUM_Q_HEADS_PER_CTA):
            # At cg2 this CTA owns heads [cta_id*NUM_Q_HEADS_PER_CTA, ...).
            var sink_head_idx = (
                cta_id * UInt32(Self.NUM_Q_HEADS_PER_CTA) + sink_row
            )
            var attn_sink_val = attn_sink_ptr.unsafe_value()[
                Int(sink_head_idx)
            ] * Float32(log2e)
            # `li` carries the +P_FP8_BIAS lift (see the P numerators); the sink
            # term must carry the SAME bias so the exp2(P_FP8_BIAS) factor
            # cancels in output_scale (MHA's fp8-bias path simply has no sink).
            output_scale = 1.0 / (
                li + exp2(attn_sink_val - mi + Self.P_FP8_BIAS)
            )
        else:
            output_scale = 1.0 / li

        var have_valid_indices = warp.vote[.uint32](
            li != Float32(0.0)
        ) != UInt32(0)
        if not have_valid_indices:
            output_scale = 1.0

        var head_row_block = UInt32(warp_idx) % UInt32(2)
        var depth_col_block = UInt32(warp_idx) // UInt32(2)
        var head_local = head_row_block * UInt32(32) + UInt32(lane_idx)

        comptime GROUP_STRIDE = Self.NUM_Q_HEADS_PER_CTA * 64
        comptime o_sw = make_swizzle[
            Self.output_dtype, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var o_ptr = kv_ptr.bitcast[Scalar[Self.output_dtype]]()

        comptime for atom_idx in range(Self.NUM_SV_ATOMS):
            comptime atom_o_tmem_addr = (
                Self.O_TMEM_ADDR + atom_idx * Self.O_ATOM_PHYS_COLS
            )
            comptime for chunk in range(2):
                comptime CHUNK = 64
                var col_group = Int(depth_col_block) * 2 + atom_idx * 4 + chunk
                var c_chunk = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=CHUNK,
                    dtype=DType.float32,
                    pack=False,
                    width=CHUNK,
                ](UInt32(atom_o_tmem_addr + chunk * CHUNK))
                tcgen05_load_wait()

                comptime for i in range(CHUNK // 2):
                    var v0_f32 = c_chunk[2 * i] * output_scale
                    var v1_f32 = c_chunk[2 * i + 1] * output_scale
                    var v = SIMD[Self.output_dtype, 2](
                        v0_f32.cast[Self.output_dtype](),
                        v1_f32.cast[Self.output_dtype](),
                    )
                    var smem_offset = col_group * GROUP_STRIDE + o_sw(
                        Int(head_local) * 64 + i * 2
                    )
                    var pad_row = Int(
                        head_local >= UInt32(Self.NUM_Q_HEADS_PER_CTA)
                    )
                    var store_off = (
                        smem_offset * (1 - pad_row)
                        + (Self.SMemType.O_SIZE - 2) * pad_row
                    )
                    (o_ptr + store_off).store[width=2](v)

        named_barrier[Int32(WARPGROUP_SIZE)](Int32(0))
        if warp_idx == 0:
            if elect_one_sync():
                fence_async_view_proxy()
                var o_smem_tile = TileTensor(
                    o_ptr,
                    row_major[Self.NUM_Q_HEADS_PER_CTA, Self.config.v_depth](),
                )
                o_tma_op.async_store(
                    o_smem_tile,
                    (
                        0,
                        Int(seq_idx) * Self.config.num_q_heads
                        + Int(cta_id) * Self.NUM_Q_HEADS_PER_CTA,
                    ),
                )
                cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()

        if warp_idx == 0:
            tcgen05_dealloc[Int32(Self.config.cta_group)](
                tmem_addr_ptr[], Self.config.sm100_tmem_cols
            )


@always_inline
def mla_prefill_sparse_qkv_fp8[
    output_dtype: DType,
    q_type: DType,
    cache_t: KVCacheT,
    config: MLASparseConfig,
    group: Int,
    q_depth: Int,
    scale_block_size: Int,
](
    output: TileTensor[output_dtype, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    scale: Float32,
    indices_stride: Int32,
    ctx: DeviceContext,
) raises:
    comptime assert q_depth == config.input_qk_depth
    comptime assert config.qk_depth == 576
    comptime assert (
        config.input_qk_depth == 576 or config.input_qk_depth == 512
    ), "sparse MLA prefill supports a 576 (rotary) or 512 (NoPE) row"
    # head=128 runs the 2-CTA (cta_group=2) f8f6f4 tile; head<=64 runs the
    # single-CTA shared-KV tile.  The two paths are separate comptime branches
    # keyed on config.cta_group throughout this file.
    comptime assert (config.cta_group == 2 and config.num_q_heads == 128) or (
        config.cta_group == 1
        and 0 < config.num_q_heads
        and config.num_q_heads <= 64
        and config.num_q_heads % 8 == 0
    ), (
        "native-fp8 sparse MLA prefill: num_q_heads must be 128 (cta_group=2)"
        " or a multiple of 8 in (0, 64] (cta_group=1)"
    )
    comptime assert config.num_kv_heads == 1
    # Operands are native FP8; the output is `config.qkv_dtype` (bf16). Q is FP8
    # (unlike the BF16-KV path where Q stays bf16), so `q_type == FP8_TYPE`.
    comptime assert q_type == FP8_TYPE
    comptime assert output_dtype == config.qkv_dtype
    # Unit-scale (scale_block_size == 0) is the DSv3.2 production path: FP8
    # latents read at scale 1, so kv_scale == v_scale == 1 and the softmax /
    # output scales match the BF16-KV kernel's unit-scale formulas. Tensorwise
    # and blockwise scale-fold are a follow-up.
    comptime assert scale_block_size == 0, (
        "native-fp8 sparse MLA prefill currently supports scale_block_size == 0"
        " (unit-scale, the DSv3.2 production path); tensorwise/blockwise"
        " scale-fold is a follow-up"
    )

    var num_q_rows = q_num_matrix_view_rows(q)
    var kv_operand = KVCacheMHAOperand(kv_cache)

    var q_tma_op = create_tensor_tile[
        Index(1, config.num_q_heads // config.cta_group, q_depth),
        swizzle_mode=SW64,
    ](ctx, q)

    # k-major SW64 FP8 gather (box_width 64, one col-group per 64 columns of the
    # row as stored).  The K producer gathers this CTA's B_TOPK/cta_group rows.
    # The cg2 V producer reuses this SAME descriptor to gather all B_TOPK keys'
    # per-CTA depth slice.
    var kv_tma_op = kv_operand.create_gather4_tma_tile[
        tile_width=config.input_qk_depth,
        tile_stride=config.input_qk_depth,
        swizzle_mode=SW64,
        tile_height=config.B_TOPK,
        tma_dtype=FP8_TYPE,
    ](ctx)

    var output_2d = TileTensor(
        output.ptr,
        row_major(num_q_rows * config.num_q_heads, config.v_depth),
    )
    var o_tma_op = create_tensor_tile[
        Index(config.num_q_heads // config.cta_group, config.v_depth),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        __desc_shape=Index(config.num_q_heads // config.cta_group, 64),
    ](ctx, output_2d)

    comptime assert type_of(topk_lengths).flat_rank == 1
    comptime assert type_of(indices).flat_rank == 1
    comptime assert topk_lengths.element_size == 1
    comptime assert indices.element_size == 1
    comptime kernel = MLAPrefillSparseQKVFP8[
        KVLUTType=type_of(kv_operand),
        output_dtype=output_dtype,
        config=config,
    ].kernel[
        type_of(topk_lengths).LayoutType,
        type_of(indices).LayoutType,
        type_of(topk_lengths).Engine,
        type_of(indices).Engine,
    ]

    comptime smem_size = size_of[MLASparseSharedMemoryQKVFP8[config]]()

    ctx.enqueue_function[kernel](
        q_tma_op,
        kv_tma_op,
        o_tma_op,
        topk_lengths,
        indices,
        kv_operand,
        scale,
        attn_sink_ptr,
        indices_stride,
        grid_dim=(config.cta_group * num_q_rows, 1, 1),
        block_dim=(config.num_threads, 1, 1),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
