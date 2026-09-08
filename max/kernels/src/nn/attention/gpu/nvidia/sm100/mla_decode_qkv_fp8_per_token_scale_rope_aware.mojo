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

"""SnapMLA FP8+BF16 MLA decode kernel for SM100 (B200).

Split content/rope kernel with per-token FP8 scaling (Steps 1+2):
- Content (nope): FP8 e4m3 for both Q_nope and K_nope (512 dims)
- Rope: BF16 for both Q_rope and K_rope (64 dims)
- P (softmax output): FP8 e4m3 reusing KV rope SMEM (P_i maps to rope stage i)
- V: FP8 e4m3 content-only (512 dims)

Uses native FP8 WGMMA (tcgen05.mma.kind::f8f6f4) for content QK and PV.
Uses BF16 WGMMA (tcgen05.mma.kind::f16) for rope QK.
Same 3-WG structure as the BF16 kernel (softmax WG, correction WG, MMA+Load+Store WG).

KV Cache Layout (640 bytes per row):
  Bytes 0-511:   FP8 content (512 dims, kv_lora_rank)
  Bytes 512-639: BF16 rope (64 dims × 2 bytes)

Per-Token FP8 Scaling (SnapMLA Approach):
  Each KV token t has a per-token scale sigma_KV[t] (one float32 value).
  In MLA's absorbed mode, K and V derive from the same latent c_KV, so
  sigma_KV[t] is shared between K and V dequantization.

  sigma_Q is per-query-token: each Q position has its own float32 scale.
  All BM=64 heads in a CTA share the same Q token, so sigma_Q is constant
  per CTA. It is folded into scale_log2e inside the Softmax function:
    scale_log2e = (1/sqrt(d_qk)) * sigma_Q[q_token_idx]

  QK scoring:  After reading combined scores S = content_raw + rope_raw from
               TMEM, each column t is multiplied by sigma_KV[t] BEFORE the
               log2e softmax scaling. This is mathematically exact under
               Scale Domain Alignment (Eq. 6): Q_rope and K_rope are
               pre-divided by their respective content scales before entering
               the kernel, so the uniform sigma application is correct.
  PV dequant:  Before writing P to FP8 SMEM, each column t of the softmax
               output is multiplied by sigma_KV[t], pre-fusing the V dequant
               scale: P'[t] = P[t] * sigma_KV[t], so PV MMA computes
               sum_t P'[t] * V_fp8[t] = sum_t P[t] * sigma_KV[t] * V_fp8[t].

MMA in QK:
  1. FP8 content MMA: Q_nope(FP8) × K_nope(FP8) → S in TMEM (c_scale=0, 8 blocks)
  2. BF16 rope MMA: Q_rope(BF16) × K_rope(BF16) → accumulate onto S (c_scale=1, 1 block)

SMEM Layout:
  Q_nope FP8:     64 × 512 × 1 = 32768 bytes   (SWIZZLE_64B)
  Q_rope BF16:    64 × 64  × 2 = 8192 bytes     (SWIZZLE_128B)
  KV content:     N × 64 × 512 × 1 bytes         (SWIZZLE_64B, N=num_kv_stages)
  KV rope:        N × 64 × 64  × 2 bytes          (SWIZZLE_128B)
  P stages:       reuses KV rope region (P_i in rope stage i; 4096B FP8 fits in 8192B BF16)
  max/li:         128 × 4 × 3 = 1536 bytes
  per-tok scales: N × 64 × 1 × 4 bytes           (float32 sigma_KV per KV token)
  barriers:       (6N+11) fixed + output barriers
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.math.constants import log2e
from std.sys import size_of
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import external_memory
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.tma_async import (
    SharedMemBarrier,
)
from layout import (
    ComptimeInt,
    CoordLike,
    Layout,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
    KVTMATile,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    SharedMemPointer,
    MBarType,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    ORaggedTMATile,
    ScalesTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
    KVPipelineGeneric,
    DecodeSM100MiscMBars,
    DecodeSProducerN,
    DecodePConsumerN,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS_Content_FP8,
    DecodeSM100QKTSS_Rope_BF16,
    DecodeSM100PVSS_FP8,
)


# ------------------------------------------------------------------------------
# SnapMLA FP8+BF16 MLA decoding kernel struct for SM100
# Content (nope) is FP8, Rope is BF16, with tensorwise FP8 scaling.
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    Engine: TensorEngine,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_per_token_scales: Bool = False,
](TrivialRegisterPassable):
    """SnapMLA FP8 decode kernel for SM100 with per-token FP8 scaling.

    Splits the MLA decode into content (FP8 e4m3, 512 dims) and rope (BF16, 64
    dims) stages, using native FP8 WGMMA for content QK and PV and BF16 WGMMA
    for rope QK. Per-token FP8 scales are folded into the softmax scale for Q
    and applied per KV token column before softmax and PV dequantization. The
    kernel runs a 3-warpgroup structure (softmax, correction, MMA+load+store)
    and supports only `NullMask` and `CausalMask`.

    Parameters:
        q_type: The precision used for the softmax accumulator and
            P-stage shared memory (`bfloat16`), even though
            content Q/K/V and P are FP8 and rope Q/K are BF16 in
            shared memory.
        KVLUTType: The paged KV cache operand type, providing the KV
            dtype and page size used for TMA loads.
        output_type: The data type of the output tensor written via
            TMA store.
        SplitAccumType: The optional pointer type for the split-K LSE
            accumulation buffer used when decoding is partitioned
            across CTAs.
        MaskType: The attention mask type; one of `NullMask` or
            `CausalMask`.
        config: The decode configuration providing tile sizes, stage
            counts, head counts, and TMEM layout for the kernel.
        ValidLengthType: The optional pointer type for the
            per-request valid sequence length buffer.
        Engine: Engine policy of the `scalar_args` tile operand.
        _is_cache_length_accurate: Whether the cache length used for
            offset computation is accurate (defaults to `False`).
        ragged: Whether ragged (variable-length) sequences are used,
            skipping blocks beyond the actual sequence length
            (defaults to `False`).
        has_per_token_scales: Whether per-token FP8 scales are loaded
            via TMA and applied to QK scoring and PV dequantization
            (defaults to `False`).
    """

    comptime kv_type = Self.KVLUTType.dtype  # float8_e4m3fn
    comptime fp8_type = DType.float8_e4m3fn
    comptime bf16_type = DType.bfloat16
    comptime AccumType = get_accum_type[Self.q_type]()

    # Number of producer arrivals for KV pipeline mbarrier:
    # Always 1 (TMA via expect_bytes only). Per-token scales are also
    # loaded via TMA on the same mbarrier, so no extra thread arrivals.
    comptime num_kv_producer = 1

    # Content dimensions: 512 (kv_lora_rank = depth)
    # Rope dimensions: 64 (qk_rope_head_dim = rope_depth)
    # Total Q depth: 576 (padded_q_depth = content + rope)

    # Content blocks: 512 / 64 = 8
    comptime NumContentBlocks = Self.config.padded_depth // Self.config.BN_QK
    # Rope blocks: 64 / 64 = 1
    comptime NumRopeBlocks = Self.config.rope_depth // Self.config.BN_QK
    # V/O blocks: 512 / 64 = 8 (V is content-only)
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK

    # FP8: 1 byte per element
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    # BF16: 2 bytes per element
    comptime bf16_bytes_per_element = size_of[Self.bf16_type]()

    # Content stage: 8 blocks * 4096 elems = 32768 FP8 elements (32768 bytes)
    comptime ContentStageElems = Self.NumContentBlocks * Self.BlockElems
    comptime ContentStageBytes = Self.ContentStageElems * Self.fp8_bytes_per_element
    # Rope stage: 1 block * 4096 elems = 4096 BF16 elements (8192 bytes)
    comptime RopeStageElems = Self.NumRopeBlocks * Self.BlockElems
    comptime RopeStageBytes = Self.RopeStageElems * Self.bf16_bytes_per_element
    # Total KV stage: 32768 + 8192 = 40960 bytes
    comptime KVStageTotalBytes = Self.ContentStageBytes + Self.RopeStageBytes

    # P stage element count (FP8): 1 block * 4096 elems = 4096
    comptime PStageElems = Self.BlockElems

    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    # Content QK MMA: FP8 (KIND_F8F6F4) — Q_nope and K_nope are FP8 in SMEM
    # BK=512 (content depth), SWIZZLE_64B
    comptime UMMAQKTSS_Content = DecodeSM100QKTSS_Content_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    # Rope QK MMA: BF16 (KIND_F16) — Q_rope and K_rope are BF16 in SMEM
    # BK=64 (rope depth), SWIZZLE_128B
    comptime UMMAQKTSS_Rope = DecodeSM100QKTSS_Rope_BF16[
        operand_type=Self.bf16_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    # PV MMA: FP8 (KIND_F8F6F4) — both P and V are FP8 in SMEM
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    comptime Common_MLA_Op = MLA_SM100_Decode_Common[
        Self.q_type,
        Self.KVLUTType,
        Self.output_type,
        Self.SplitAccumType,
        Self.MaskType,
        Self.config,
        Self.ValidLengthType,
        Self._is_cache_length_accurate,
        Self.ragged,
    ]

    # Number of pipeline stages for KV, S, and P (dynamically computed, typically 4)
    comptime num_stages = Self.config.num_kv_stages

    # --------------------------------------------------------------------------
    # Main kernel function — 3 Warpgroups, 384 threads
    # --------------------------------------------------------------------------
    #    Softmax WG (warps 0-3), Correction WG (warps 4-7),
    #    MMA+Load+Store WG (warps 8-11)
    #    Warp assignments within WG2: warp 8 = Load, warp 9 = MMA QK,
    #                                 warp 10 = MMA PV, warp 11 = Store
    #
    # SMEM layout (split content FP8 + rope BF16, N stages):
    #   Q_nope FP8:    64 × 512 × 1 = 32768 bytes (SWIZZLE_64B)
    #   Q_rope BF16:   64 × 64  × 2 = 8192 bytes  (SWIZZLE_128B)
    #   KV content:    N × 64 × 512 × 1 bytes (SWIZZLE_64B)
    #   KV rope:       N × 64 × 64  × 2 bytes (SWIZZLE_128B)
    #   P stages:      reuses KV rope (P_i in rope stage i; 4096B FP8 ⊂ 8192B BF16)
    #   max/li:        128 × 4 × 3 = 1536 bytes
    #   per-tok scales: N × 64 × 1 × 4 = N × 256 bytes (float32 sigma_KV per KV token)
    #   barriers:      (6N+11) fixed + output barriers
    #
    # The FP8 dequant scale is folded into the softmax scale:
    #   qk_scale = (1.0 / sqrt(d_qk)) * kv_scale
    # --------------------------------------------------------------------------

    @staticmethod
    @__llvm_arg_metadata(q_nope_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(q_rope_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_content_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(scale_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__name(
        t"sm100_mla_decode_qkv_fp8_per_token_scale_rope_aware_{Self.fp8_type}_{Self.bf16_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        # Q_nope TMA: FP8, 64×512, SWIZZLE_64B
        q_nope_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,  # 64
            BK=Self.config.padded_depth,  # 512
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ],
        # Q_rope TMA: BF16, 64×64, SWIZZLE_128B
        q_rope_tma: QOTMATile[
            dtype=Self.bf16_type,
            BM=Self.config.BM,  # 64
            BK=Self.config.rope_depth,  # 64
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ],
        # K_content TMA: FP8, 64×512, SWIZZLE_64B
        k_content_tma: KVTMATile[
            dtype=Self.fp8_type,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            BN=Self.config.BK_PV,  # 64
            BK=Self.config.padded_depth,  # 512
        ],
        # K_rope TMA: BF16, 64×64, SWIZZLE_128B
        k_rope_tma: KVTMATile[
            dtype=Self.bf16_type,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            BN=Self.config.BK_PV,  # 64
            BK=Self.config.rope_depth,  # 64
        ],
        # Per-token scales TMA: float32, [1, BN_QK], SWIZZLE_NONE
        scale_tma: ScalesTMATile[BN_QK=Self.config.BN_QK],
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            # Per-warp output stripe (= BN_PV/4), not BN_QK.
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType=Self.ValidLengthType,
            MaskType=Self.MaskType,
            SplitAccumType=Self.SplitAccumType,
        ],
        # Per-token Q scale pointer: float32 array with one scale per Q token.
        # sigma_Q[q_token_idx] is folded into scale_log2e inside Softmax.
        # Null pointer means no Q scale (sigma_Q = 1.0).
        q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]],
        scalar_args: TileTensor[
            .int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ],
    ):
        # SlidingWindowCausalMask is supported ONLY by the native FP8 backend
        # (MLA_SM100_Decode_QKV_FP8). Reject it here at comptime.
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_QKV_FP8_PerTokenScale_RopeAware only supports"
            " NullMask and CausalMask. Sliding window is supported only by"
            " MLA_SM100_Decode_QKV_FP8 (native FP8)."
        )

        # Extract scalar launch args from the stable device buffer.
        var batch_size = Int(scalar_args.raw_load(0))
        var q_max_seq_len = Int(scalar_args.raw_load(1))
        var num_partitions = mla_decode_pack.num_partitions

        # Register allocation for 3 WGs (Softmax, Correction, MMA+Load+Store).
        #
        # Per-token FP8 scaling caches 32 float32 sigma_KV values in registers
        # inside Softmax (_sigma_kv_regs), requiring +32 regs vs the baseline
        # (192). We give Softmax 224 regs by having both Correction and
        # MMA+Load+Store donate registers via warpgroup_reg_dealloc.
        #
        # Assuming compiler initial X=168 regs/thread:
        #   Softmax inc 224: gains (224-168)*128 = 7168 regs from pool
        #   Correction dec 152: donates (168-152)*128 = 2048 regs to pool
        #   MMA+Load dec 112: donates (168-112)*128 = 7168 regs to pool
        #   Pool balance: 2048+7168 = 9216 donated >= 7168 claimed. OK.
        #
        # Correction only needs ~84 regs (64 for O tile + overhead);
        # the blockwise FP8 kernel runs Correction with just 72.
        comptime num_reg_softmax = 224
        comptime num_reg_correction = 152
        comptime num_reg_other = 112
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ](
            kv_lut,
            rebind[
                UnsafePointer[
                    Scalar[Self.ValidLengthType.dtype],
                    ImmutAnyOrigin,
                    address_space=.GENERIC,
                ]
            ](valid_length.value()),
            q_max_seq_len,
            num_partitions,
            batch_size,
        )

        # Early exit for split-K: CTAs with no work
        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )
                return

        # Early exit for ragged: skip blocks beyond actual sequence length
        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    Self.Common_MLA_Op.pdl_early_exit(
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )

                return

        # ---- SMEM layout (split content FP8 + rope BF16, N stages) ----
        # All SMEM starts from a single base pointer (dynamic shared memory).
        # We use byte-level pointer arithmetic via UInt8 bitcasts.

        # Q_nope FP8: 64 × 512 × 1 = 32768 bytes (SWIZZLE_64B)
        var q_nope_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        # Q_rope BF16: starts right after Q_nope
        # 64 × 64 × 2 = 8192 bytes (SWIZZLE_128B)
        var q_rope_smem = (q_nope_smem + Self.ContentStageElems).bitcast[
            Scalar[Self.bf16_type]
        ]()

        # KV content SMEM: N stages of 64 × 512 FP8
        # Starts after Q_nope + Q_rope
        # Q total bytes = 32768 + 8192 = 40960
        # In FP8 element count: 40960 / 1 = 40960
        comptime q_total_fp8_elems = Self.ContentStageElems + Self.RopeStageElems * Self.bf16_bytes_per_element // Self.fp8_bytes_per_element
        var kv_content_smem = q_nope_smem + q_total_fp8_elems

        # KV rope SMEM: N stages of 64 × 64 BF16
        # Starts after KV content stages
        comptime kv_total_stages = Self.config.num_kv_stages
        var kv_rope_smem = (
            kv_content_smem + Self.ContentStageElems * kv_total_stages
        ).bitcast[Scalar[Self.bf16_type]]()

        # P reuses KV rope SMEM: P_i maps to rope stage i.
        # This optimization is specific to per_token_scale_rope_aware.
        # The BF16 and other FP8 kernels (blockscale, tensorscale) still use
        # separate P SMEM regions. Here, rope is consumed by QK MMA (warp 9)
        # BEFORE softmax produces P, and KV stage barriers prevent the Load
        # warp from overwriting until PV MMA (warp 10) finishes.
        # P (4096 bytes FP8) fits inside rope (8192 bytes BF16).
        # Stride between P stages = one rope stage in FP8 elements.
        var p_smem = kv_rope_smem.bitcast[Scalar[Self.fp8_type]]()
        # P stage stride in FP8 elements: RopeStageBytes / fp8_bytes = 8192 / 1 = 8192
        comptime p_stage_stride_fp8_elems = Self.RopeStageElems * Self.bf16_bytes_per_element // Self.fp8_bytes_per_element

        # Output SMEM reuses KV content SMEM (same as BF16 kernel)
        var out_smem_start = kv_content_smem
        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        # max_smem/li_smem: placed after KV rope stages (P reuses rope, no gap)
        # max_smem is double-buffered (2 × 128 elements) to avoid a race
        # condition in softmax; li_smem is a single 128-element buffer.
        var max_smem = (
            kv_rope_smem + Self.RopeStageElems * kv_total_stages
        ).bitcast[Scalar[Self.AccumType]]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        # Per-token scale SMEM: N stages × 256 bytes each (64 tokens × 1 × float32).
        # Placed right after li_smem. Used by the load warp to store per-token
        # sigma_KV values from HBM before they are applied later.
        # In MLA's absorbed mode, K and V share one scale per token.
        var scale_smem_base = (li_smem + WARPGROUP_SIZE).bitcast[Float32]()
        comptime per_token_scales_total_elems = Self.config.num_kv_stages * Self.config.per_token_scales_per_stage // size_of[
            DType.float32
        ]()

        # ---- Barrier layout (6N+11 fixed for N-stage pipelines) ----
        # bar_q(1) + kv(2N) + s(2N) + p(2N) + o(4) + c(2) + corr_done(4)
        var mbar_base: MBarType = (
            (scale_smem_base + per_token_scales_total_elems)
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_base: MBarType = mbar_base + 1

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.num_kv_producer,
            num_consumer=2,
        ](mbar_kv_base)

        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()
        # S pipeline: N-stage (matching KV stages)
        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = s_bars.end()
        # P pipeline: N-stage (matching KV stages)
        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base = p_bars.end()
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](mbar_base)
        mbar_base = o_bars.end()
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = c_bars.end()
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = corr_done_bars.end()
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base += out_pipeline.num_mbars()

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        var is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                corr_done_bars.init()
                q_nope_tma.prefetch_descriptor()
                q_rope_tma.prefetch_descriptor()
                k_content_tma.prefetch_descriptor()
                k_rope_tma.prefetch_descriptor()
                scale_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()
            # sigma_Q folding is done inside Softmax (per-token, using
            # q_scale_ptr[q_token_idx] from OffsetPosition).
            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages=Self.num_stages,
                fp8_p_stage_stride=p_stage_stride_fp8_elems,
                has_per_token_scales=Self.has_per_token_scales,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.bitcast[
                    Scalar[Self.Common_MLA_Op.q_type]
                ]().as_unsafe_any_origin(),
                max_smem.as_unsafe_any_origin(),
                li_smem.as_unsafe_any_origin(),
                out_smem.as_unsafe_any_origin(),
                c_bars,
                corr_done_bars,
                out_pipeline,
                offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                scale_k_smem=scale_smem_base.unsafe_origin_cast[MutAnyOrigin](),
                q_scale_ptr=q_scale_ptr,
            )
        elif warp_idx >= 4 and warp_idx < 8:  # correction warpgroup
            warpgroup_reg_dealloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_nope_tma,
                    q_rope_tma,
                    k_content_tma,
                    k_rope_tma,
                    scale_tma,
                    kv_lut,
                    q_nope_smem.as_unsafe_any_origin(),
                    q_rope_smem.as_unsafe_any_origin(),
                    kv_content_smem.as_unsafe_any_origin(),
                    kv_rope_smem.as_unsafe_any_origin(),
                    mbar_q,
                    kv_pipeline,
                    offset_position,
                    scale_smem_base.as_unsafe_any_origin(),
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_nope_smem.as_unsafe_any_origin(),
                    q_rope_smem.as_unsafe_any_origin(),
                    kv_content_smem.as_unsafe_any_origin(),
                    kv_rope_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_content_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 11:
                Self.Common_MLA_Op.store(
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        barrier()

        # PDL: Signal that this CTA is done
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # --------------------------------------------------------------------------
    # Load: TMA Q_nope (FP8) + Q_rope (BF16), TMA K_content (FP8) + K_rope (BF16)
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def load(
        q_nope_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.padded_depth,  # 512
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
        ],
        q_rope_tma: QOTMATile[
            dtype=Self.bf16_type,
            BM=Self.config.BM,
            BK=Self.config.rope_depth,  # 64
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        ],
        k_content_tma: KVTMATile[
            dtype=Self.fp8_type,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_64B,
            BN=Self.config.BK_PV,  # 64
            BK=Self.config.padded_depth,  # 512
        ],
        k_rope_tma: KVTMATile[
            dtype=Self.bf16_type,
            swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
            BN=Self.config.BK_PV,  # 64
            BK=Self.config.rope_depth,  # 64
        ],
        scale_tma: ScalesTMATile[BN_QK=Self.config.BN_QK],
        kv_lut: Self.KVLUTType,
        q_nope_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        q_rope_smem: SharedMemPointer[Scalar[Self.bf16_type]],
        kv_content_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_rope_smem: SharedMemPointer[Scalar[Self.bf16_type]],
        mbar_q: MBarType,
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.num_kv_producer,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
        scale_smem_base: SharedMemPointer[Float32],
    ):
        if offset_position.num_keys_this_split == 0:
            return

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        # Alignment of `kv_row` produced by mask-driven iteration.
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.config.BM, Self.config.BN_QK, Self.KVLUTType.page_size
        ]()

        # We manage the KV pipeline manually for barrier sync,
        # but compute SMEM pointers ourselves for the split layout.
        var kv_prod = DecodeKVProducer[
            Self.fp8_type, Self.config, Self.num_kv_producer
        ](kv_pipeline, kv_content_smem)
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
            UInt32(offset_position.batch_idx), kv_row
        )
        # For the scale TMA (flat 2D layout), we still need a single
        # base row index. `paged_rows.rows[0]` matches the old
        # `kv_lut.row_idx(...)` result. The scale TMA assumes
        # `page_size >= BN_QK` (tokens within a tile are contiguous in the
        # global scales array); small pages are not supported in this
        # kernel.
        var kv_gmem_row: UInt32 = UInt32(paged_rows.rows[0])

        # Q bytes: content FP8 (BM*512*1) + rope BF16 (BM*64*2)
        comptime q_content_bytes = Self.config.BM * Self.config.depth * Self.fp8_bytes_per_element
        comptime q_rope_bytes = Self.config.BM * Self.config.rope_depth * Self.bf16_bytes_per_element
        # KV bytes per tile: content FP8 (BN_QK*512*1) + rope BF16 (BN_QK*64*2)
        comptime kv_content_bytes = Self.config.BN_QK * Self.config.depth * Self.fp8_bytes_per_element
        comptime kv_rope_bytes = Self.config.BN_QK * Self.config.rope_depth * Self.bf16_bytes_per_element
        # Scale bytes per tile: BN_QK * 1 * sizeof(float32) = 256 bytes
        comptime scale_bytes = Self.config.BN_QK * 4
        # Each scale stage holds BN_QK float32 values = 64 elements
        comptime scale_elems_per_stage = Self.config.BN_QK

        # TMA only uses .ptr — flat row_major TileTensor is sufficient.
        comptime _smem_tt[dtype: DType, elems: Int] = TileTensor[
            dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=.SHARED,
        ]
        comptime q_nope_elems = type_of(q_nope_tma).tile_shape[0] * type_of(
            q_nope_tma
        ).tile_shape[1]
        comptime q_rope_elems = type_of(q_rope_tma).tile_shape[0] * type_of(
            q_rope_tma
        ).tile_shape[1]
        comptime scale_elems = type_of(scale_tma).tile_shape[0] * type_of(
            scale_tma
        ).tile_shape[1]

        # Load Q: Q_nope (FP8) and Q_rope (BF16) on the same barrier
        if is_leader:
            mbar_q[].expect_bytes(Int32(q_content_bytes + q_rope_bytes))
            # Q_nope TMA: load FP8 content Q into q_nope_smem
            var q_nope_tensor = _smem_tt[Self.fp8_type, q_nope_elems](
                q_nope_smem, tt_row_major[q_nope_elems]()
            )
            q_nope_tma.async_copy(q_nope_tensor, mbar_q[], (0, row))
            # Q_rope TMA: load BF16 rope Q into q_rope_smem
            var q_rope_tensor = _smem_tt[Self.bf16_type, q_rope_elems](
                q_rope_smem, tt_row_major[q_rope_elems]()
            )
            q_rope_tma.async_copy(q_rope_tensor, mbar_q[], (0, row))

        # Load first KV tile: content + rope + scales on the same barrier.
        # All three TMA copies share one expect_bytes call, so the mbar
        # fires only after all data (content, rope, and scales) has landed.
        var k0_bar: MBarType = kv_prod.producer_mbar[qk_stage=0]()
        var stage0_idx = kv_prod.stage_index[qk_stage=0]()

        if is_leader:
            comptime if Self.has_per_token_scales:
                k0_bar[].expect_bytes(
                    Int32(kv_content_bytes + kv_rope_bytes + scale_bytes)
                )
            else:
                k0_bar[].expect_bytes(Int32(kv_content_bytes + kv_rope_bytes))
            # K_content TMA: load FP8 content into kv_content_smem
            var content_stage_ptr = kv_content_smem + stage0_idx * UInt32(
                Self.ContentStageElems
            )
            paged_rows.tma_copy_k[needs_partial=False](
                k_content_tma,
                content_stage_ptr,
                k0_bar[],
                kv_head_idx=UInt32(0),
                elect=elect_mask,
            )
            # K_rope TMA: load BF16 rope into kv_rope_smem
            var rope_stage_ptr = kv_rope_smem + stage0_idx * UInt32(
                Self.RopeStageElems
            )
            paged_rows.tma_copy_k[needs_partial=False](
                k_rope_tma,
                rope_stage_ptr,
                k0_bar[],
                kv_head_idx=UInt32(0),
                elect=elect_mask,
            )
            # Scale TMA: load BN_QK float32 per-token scales into scale SMEM.
            # The scale TMA treats scales as a flat [1, total_elements] 2D
            # tensor; the column coordinate is the physical row index
            # (kv_gmem_row) which directly indexes the flat scales array.
            comptime if Self.has_per_token_scales:
                var scale_stage_ptr = scale_smem_base + stage0_idx * UInt32(
                    scale_elems_per_stage
                )
                var scale_tensor = _smem_tt[.float32, scale_elems](
                    scale_stage_ptr, tt_row_major[scale_elems]()
                )
                scale_tma.async_copy(
                    scale_tensor,
                    k0_bar[],
                    (Int(kv_gmem_row), 0),
                )

        kv_prod.commit_step()
        kv_row += UInt32(Self.config.BN_QK)

        # Load remaining KV tiles
        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_prod.acquire[qk_stage=0]()
            var stage_idx = kv_prod.stage_index[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()
            kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
            var paged_rows = kv_lut.populate[Self.config.BN_QK, base_alignment](
                UInt32(offset_position.batch_idx), kv_row
            )
            var kv_gmem_row: UInt32 = UInt32(paged_rows.rows[0])

            if is_leader:
                comptime if Self.has_per_token_scales:
                    k_mbar[].expect_bytes(
                        Int32(kv_content_bytes + kv_rope_bytes + scale_bytes)
                    )
                else:
                    k_mbar[].expect_bytes(
                        Int32(kv_content_bytes + kv_rope_bytes)
                    )
                # K_content TMA
                var content_stage_ptr = kv_content_smem + stage_idx * UInt32(
                    Self.ContentStageElems
                )
                paged_rows.tma_copy_k[needs_partial=False](
                    k_content_tma,
                    content_stage_ptr,
                    k_mbar[],
                    kv_head_idx=UInt32(0),
                    elect=elect_mask,
                )
                # K_rope TMA
                var rope_stage_ptr = kv_rope_smem + stage_idx * UInt32(
                    Self.RopeStageElems
                )
                paged_rows.tma_copy_k[needs_partial=False](
                    k_rope_tma,
                    rope_stage_ptr,
                    k_mbar[],
                    kv_head_idx=UInt32(0),
                    elect=elect_mask,
                )
                # Scale TMA
                comptime if Self.has_per_token_scales:
                    var scale_stage_ptr = scale_smem_base + stage_idx * UInt32(
                        scale_elems_per_stage
                    )
                    var scale_tensor = _smem_tt[.float32, scale_elems](
                        scale_stage_ptr, tt_row_major[scale_elems]()
                    )
                    scale_tma.async_copy(
                        scale_tensor,
                        k_mbar[],
                        (Int(kv_gmem_row), 0),
                    )

            kv_row += UInt32(Self.config.BN_QK)
            kv_prod.commit_step()
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA QK: Q_nope(FP8) × K_nope(FP8) + Q_rope(BF16) × K_rope(BF16) → S(TMEM)
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def mmaQK(
        tmem_addr: UInt32,
        q_nope_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        q_rope_smem: SharedMemPointer[Scalar[Self.bf16_type]],
        kv_content_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_rope_smem: SharedMemPointer[Scalar[Self.bf16_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.num_kv_producer,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[
            Self.fp8_type, Self.config, Self.num_kv_producer
        ](kv_pipeline, kv_content_smem)
        # N-stage S producer
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        # Content FP8 descriptors via UMMAQKTSS_Content struct
        var q_content_desc = Self.UMMAQKTSS_Content.descriptor_q_block(
            q_nope_smem
        )
        var k_content_desc = Self.UMMAQKTSS_Content.descriptor_k_block(
            kv_content_smem
        )

        # Rope BF16 descriptors via UMMAQKTSS_Rope struct
        var q_rope_desc = Self.UMMAQKTSS_Rope.descriptor_q_block(q_rope_smem)
        var k_rope_desc = Self.UMMAQKTSS_Rope.descriptor_k_block(kv_rope_smem)

        # Stage strides in bytes for content and rope
        comptime content_stage_stride_bytes = Self.ContentStageBytes
        comptime rope_stage_stride_bytes = Self.RopeStageBytes

        mbar_q[].wait(0)
        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            s_prod.acquire()

            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            kv_cons.wait[qk_stage=0]()
            var k_slot_index = kv_cons.stage_index[qk_stage=0]()

            # --- FP8 content MMA: Q_nope × K_nope → S (c_scale=0, overwrite) ---
            Self.UMMAQKTSS_Content.mma[stage_idx=0](
                a=q_content_desc,
                b=k_content_desc
                + k_slot_index * UInt32(content_stage_stride_bytes),
                c=s_tmem_slot,
                c_scale=UInt32(0),  # c_scale=0: overwrite
                elect=elect_mask,
            )

            # --- BF16 rope MMA: Q_rope × K_rope → accumulate onto S (c_scale=1) ---
            Self.UMMAQKTSS_Rope.mma[stage_idx=0](
                a=q_rope_desc,
                b=k_rope_desc + k_slot_index * UInt32(rope_stage_stride_bytes),
                c=s_tmem_slot,
                c_scale=UInt32(1),  # c_scale=1: accumulate
                elect=elect_mask,
            )

            tcgen05_fence_before()
            s_prod.commit_mma(elect_mask)
            kv_cons.release[qk_stage=0](elect_mask)
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA PV: P(FP8) × V(FP8) → O(TMEM)
    # V is content-only (512 dims FP8), stored in kv_content_smem.
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def mmaPV(
        tmem_addr: UInt32,
        kv_content_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.num_kv_producer,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )

        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[
            Self.fp8_type, Self.config, Self.num_kv_producer
        ](kv_pipeline, kv_content_smem)
        # N-stage P consumer
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        # P descriptor: P reuses KV rope SMEM (P_i maps to rope stage i)
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        # V descriptor: points to the KV content SMEM (V = content-only, 512 dims)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_content_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        # Content stage stride in bytes (for V data)
        comptime kv_content_stage_stride_bytes = Self.ContentStageBytes
        # P stage stride in bytes: one rope stage = RopeStageBytes (8192)
        # P_i starts at rope_base + i * RopeStageBytes
        comptime p_stage_stride_in_bytes = Self.RopeStageBytes
        comptime block_stride_in_bytes = Self.BlockElems * Self.fp8_bytes_per_element

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            # PV uses content-only V (512 dims = 8 blocks of 64)
            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_content_stage_stride_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN_QK // 2),
                    c_scale=c_scale,
                    elect=elect_mask,
                )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            kv_cons.release[qk_stage=0](elect_mask)
            tcgen05_fence_before()

            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
