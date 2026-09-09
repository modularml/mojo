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
"""Dispatch logic for grouped 1D-1D block-scaled SM100 matmul.

Selects optimal kernel configuration based on (N, K) shape and workload
size, with parameters tuned via ablation on B200. NVFP4 gets shape-tuned
three-regime dispatch; MXFP4 and MXFP8 use default configs.

When `override=True`, uses the caller's AB_swapped/mma_bn/cta_group/
num_pipeline_stages directly (for ablation studies and benchmarking).
When `override=False` (default), ignores those parameters and selects
from the tuning table based on (N, K) and `estimated_total_m`.

NVFP4 routing (B200-tuned via ablation):

  (N=512, K=7168) Kimi K2.5 TP=8 up-proj: shape-specific 2-branch override.
    Phase-2 ablation showed regime classifier picks suboptimal
    (mma_bn, cta_group) here; all regimes converge on cta_group=2, stages=6,
    with mma_bn=64 for decode (avg_m <= 8) and mma_bn=128 otherwise.

  Other shapes: three-regime classifier keyed on
  avg_m = estimated_total_m / num_active_experts.

    Decode (avg_m <= 8):           AB_swapped=True, mma_bn=8,  cta_group=1
    Small prefill (avg_m <= 64):   AB_swapped=True, mma_bn=64, cta_group=2
    Large prefill (avg_m > 64):    AB_swapped=True, mma_bn=128, cta_group=2

  Tuned stages per (N, K) live at the three `_dispatch_regime` call sites:
    (N=4096, K=7168) DeepSeek-V3 up-proj,
    (N=7168, K=2048) DeepSeek-V3 down-proj,
    (N=7168, K=256) Kimi K2.5 TP=8 down-proj (large prefill only).
  Unknown shapes fall through to stages=auto.
"""

from std.collections import Optional

from max.gpu.host import DeviceContext
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from max.gpu.primitives.grid_controls import PDLLevel
from std.utils.index import Index
from layout import TileTensor

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    block_scaled_umma_kind,
    is_w4a8_operand_pair,
)
from structured_kernels.trace_buf import NullTrace, TraceBuf
from ..structured_kernels.config import BlockScaledMatmulConfig, GEMMKind
from .grouped_1d1d_matmul import grouped_matmul_block_scaled
from .grouped_1d1d_matmul_kernel import (
    NullSwiGLUOutput,
    SwiGLUOutput,
)


# Regime thresholds on avg_m = estimated_total_m / num_active_experts. Rows
# (avg_m <= DECODE_AVG_M) pick the decode config; rows in
# (DECODE_AVG_M, SMALL_PREFILL_AVG_M] pick small prefill; rest pick large
# prefill. Tuned on B200 NVFP4 traffic; don't change without a new ablation.
comptime DECODE_AVG_M = 8
comptime SMALL_PREFILL_AVG_M = 64


def _launch_grouped_block_scaled[
    transpose_b: Bool,
    AB_swapped: Bool,
    mma_bn: Int,
    cta_group: Int,
    num_pipeline_stages: Optional[Int] = None,
    scaling_kind: UMMAKind = UMMAKind.KIND_MXF4NVF4,
    pdl_level: PDLLevel = PDLLevel.ON,
    fuse_swiglu: Bool = False,
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput[],
    swiglu_match_bf16: Bool = True,
    swiglu_disable_compute: Bool = False,
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    swiglu_use_inplace: Bool = False,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
    swiglu_out: SwiGLUOutputT = NullSwiGLUOutput[](),
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Build config and launch grouped block-scaled matmul kernel.

    Parameters:
        transpose_b: Whether B is transposed.
        AB_swapped: Whether A/B operands are swapped.
        mma_bn: MMA tile N dimension.
        cta_group: CTA group size.
        num_pipeline_stages: Pipeline depth override. None = auto-compute.
        scaling_kind: Block-scaling format (NVFP4, MXFP4, or MXFP8).
        pdl_level: Programmatic dependent launch level.
        fuse_swiglu: When True, the kernel fuses SwiGLU + block-scaled
            quant (NVFP4 or MXFP8) into the epilogue. Caller pre-permutes
            W on the N axis with sigma so adjacent N columns hold the
            (gate, up) pair.
        SwiGLUOutputT: Trait impl for the fused-output sink. Defaults to
            zero-sized `NullSwiGLUOutput` when `fuse_swiglu=False`.
        swiglu_match_bf16: When True, the SwiGLU epilogue applies an
            fp32 -> bf16 -> fp32 round-trip on the SMEM scratchpad to
            match the chained reference's BF16 GMEM precision.
        swiglu_disable_compute: Diagnostic only. When True, the
            cooperative SwiGLU+quant body is skipped while keeping the
            EPI structure intact. Output is invalid; do not use in
            production.
        swiglu_enable_trace: When True, the kernel records per-tile
            timestamps into `trace_buf`. DCEs to a no-op when False.
        TraceBufT: Trait impl for the per-CTA timestamp buffer.
        swiglu_use_inplace: When True, the fused epilogue uses the
            in-place register path (no SMEM scratch) on decode shapes.
            Default False keeps the cooperative path.
    """
    # `grouped_matmul_block_scaled` feeds the weights into the kernel's A slot
    # when `AB_swapped`, so the config's per-operand dtypes have to follow the
    # tensors. Invisible until the two dtypes differ, which is W4A8.
    comptime a_type = b.dtype if AB_swapped else a.dtype
    comptime b_type = a.dtype if AB_swapped else b.dtype
    comptime c_type = c.dtype
    comptime sfa_dtype = a_scales.dtype
    comptime sfb_dtype = b_scales.dtype
    comptime umma_shape = Index(128 * cta_group, mma_bn, 32)

    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=Index(cta_group, 1, 1),
        mma_shape=umma_shape,
        block_swizzle_size=8,
        cta_group=cta_group,
        AB_swapped=AB_swapped,
        k_group_size=1,
        num_pipeline_stages=num_pipeline_stages,
        num_accum_pipeline_stages=1 if mma_bn > 128 else 2,
        is_gmm=True,
        gemm_kind=GEMMKind.GMM,
    )

    # Gate the in-place register-only epilogue path: it wins on the
    # small-BN / decode regime (mma_bn <= 8) where the SMEM scratchpad
    # round-trip dominates, but loses on the prefill regime (mma_bn >=
    # 64) because the per-tile cross-lane shuffles cost more than the
    # cooperative loop's SMEM-amortized work pattern. The user-level
    # opt-in (`swiglu_use_inplace=True`) is honored only on shapes the
    # path is known to win on.
    comptime _swiglu_use_inplace = swiglu_use_inplace and (mma_bn <= 8)

    grouped_matmul_block_scaled[
        transpose_b=transpose_b,
        config=config,
        pdl_level=pdl_level,
        fuse_swiglu=fuse_swiglu,
        SwiGLUOutputT=SwiGLUOutputT,
        swiglu_match_bf16=swiglu_match_bf16,
        swiglu_disable_compute=swiglu_disable_compute,
        swiglu_enable_trace=swiglu_enable_trace,
        TraceBufT=TraceBufT,
        swiglu_use_inplace=_swiglu_use_inplace,
    ](
        c,
        a,
        a_offsets,
        a_scale_offsets,
        b,
        expert_ids,
        a_scales,
        b_scales,
        expert_scales,
        num_active_experts,
        ctx,
        swiglu_out,
        trace_buf,
    )


def _dispatch_regime[
    transpose_b: Bool,
    N: Int,
    K: Int,
    mma_bn: Int,
    cta_group: Int,
    stages_4096_7168: Optional[Int],
    stages_7168_2048: Optional[Int],
    stages_7168_256: Optional[Int],
    pdl_level: PDLLevel = PDLLevel.ON,
    fuse_swiglu: Bool = False,
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput[],
    swiglu_match_bf16: Bool = True,
    swiglu_disable_compute: Bool = False,
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    swiglu_use_inplace: Bool = False,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
    swiglu_out: SwiGLUOutputT = NullSwiGLUOutput[](),
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Dispatch with shape-specific pipeline stages.

    Uses tuned stages for known (N, K) shapes, auto-computes for others.
    """
    comptime stages = (
        stages_4096_7168 if (N == 4096 and K == 7168) else stages_7168_2048 if (
            N == 7168 and K == 2048
        ) else stages_7168_256 if (N == 7168 and K == 256) else Optional[Int](
            None
        )
    )
    _launch_grouped_block_scaled[
        transpose_b,
        True,
        mma_bn,
        cta_group,
        num_pipeline_stages=stages,
        pdl_level=pdl_level,
        fuse_swiglu=fuse_swiglu,
        SwiGLUOutputT=SwiGLUOutputT,
        swiglu_match_bf16=swiglu_match_bf16,
        swiglu_disable_compute=swiglu_disable_compute,
        swiglu_enable_trace=swiglu_enable_trace,
        TraceBufT=TraceBufT,
        swiglu_use_inplace=swiglu_use_inplace,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        a_scale_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        ctx,
        swiglu_out,
        trace_buf,
    )


def grouped_matmul_nvfp4_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    override: Bool = False,
    AB_swapped: Bool = True,
    mma_bn: Int = 8,
    cta_group: Int = 1,
    num_pipeline_stages: Int = -1,
    pdl_level: PDLLevel = PDLLevel.ON,
    fuse_swiglu: Bool = False,
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput[],
    swiglu_match_bf16: Bool = True,
    swiglu_disable_compute: Bool = False,
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    swiglu_use_inplace: Bool = False,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    swiglu_out: SwiGLUOutputT = NullSwiGLUOutput[](),
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Dispatch grouped NVFP4 matmul with shape-tuned configuration.

    When override=False (default, production), selects kernel parameters
    from the tuning table keyed on (N, K). The caller's
    AB_swapped/mma_bn/cta_group/num_pipeline_stages are ignored.

    When override=True (ablation/benchmarking), uses the caller's
    parameter values directly. num_pipeline_stages=-1 = auto-compute.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).
        override: If True, use caller's config params directly.
            If False, use tuning table.
        AB_swapped: A/B swap (only used when override=True).
        mma_bn: MMA tile N dimension (only used when override=True).
        cta_group: CTA group size (only used when override=True).
        num_pipeline_stages: Pipeline depth (only used when override=True).
            -1 = auto-compute.
        pdl_level: Programmatic dependent launch level.
        fuse_swiglu: When True, the kernel fuses SwiGLU + NVFP4 quant
            into the epilogue (caller pre-permutes W with σ).
        SwiGLUOutputT: Trait impl for the fused-output sink. Defaults to
            zero-sized `NullSwiGLUOutput` when `fuse_swiglu=False`.
        swiglu_match_bf16: When True, the SwiGLU+NVFP4 epilogue applies an
            fp32 -> bf16 -> fp32 round-trip on the SMEM scratchpad to match
            the chained reference's BF16 GMEM precision (byte-exact).
        swiglu_disable_compute: Diagnostic only. When True, the cooperative
            SwiGLU+quant body is skipped while keeping the EPI structure
            intact. Output is invalid; do not use in production.
        swiglu_enable_trace: When True, the kernel records per-tile
            timestamps into `trace_buf`. DCEs to a no-op when False.
        TraceBufT: Trait impl for the per-CTA timestamp buffer. Defaults to
            zero-sized `NullTrace` when `swiglu_enable_trace=False`.
        swiglu_use_inplace: When True, the fused SwiGLU+NVFP4 epilogue
            uses the in-place register-only path (no SMEM scratch).
            Default False keeps the original scatter+cooperative path.

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K//2 packed).
        b: Weight tensor B (num_experts, N, K//2 packed).
        a_scales: Scale factors for A (5D).
        b_scales: Scale factors for B (6D).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
        swiglu_out: Sink carrier when `fuse_swiglu=True` (packed
            NVFP4 + E4M3 SF tile). `NullSwiGLUOutput()` otherwise.
        trace_buf: Per-CTA timestamp buffer when `swiglu_enable_trace=True`.
            `NullTrace()` otherwise.
    """
    comptime if override:
        # Ablation/benchmarking: use caller's explicit parameters.
        comptime _stages = Optional[Int](
            num_pipeline_stages
        ) if num_pipeline_stages > 0 else Optional[Int](None)
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped,
            mma_bn,
            cta_group,
            num_pipeline_stages=_stages,
            pdl_level=pdl_level,
            fuse_swiglu=fuse_swiglu,
            SwiGLUOutputT=SwiGLUOutputT,
            swiglu_match_bf16=swiglu_match_bf16,
            swiglu_disable_compute=swiglu_disable_compute,
            swiglu_enable_trace=swiglu_enable_trace,
            TraceBufT=TraceBufT,
            swiglu_use_inplace=swiglu_use_inplace,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
            swiglu_out,
            trace_buf,
        )
    else:
        # Production: tuning table keyed on (N, K).
        comptime N = type_of(c).static_shape[1]
        comptime packed_K = type_of(a).static_shape[1]
        comptime K = packed_K * 2  # NVFP4: 2 values per byte

        # Nested forwarder: every regime threads the SAME 13 runtime args
        # and the SAME forwarding comptime params (pdl_level, fuse_swiglu,
        # the swiglu knobs, the trace knobs) to `_dispatch_regime`; only
        # (mma_bn, cta_group, stages) vary. Factoring the call here keeps
        # the regime selection below a one-liner per regime.
        @always_inline
        @__parameter
        def _regime[
            mma_bn: Int,
            cta_group: Int,
            stages_4096_7168: Optional[Int],
            stages_7168_2048: Optional[Int],
            stages_7168_256: Optional[Int],
        ]() raises:
            _dispatch_regime[
                transpose_b,
                N,
                K,
                mma_bn=mma_bn,
                cta_group=cta_group,
                stages_4096_7168=stages_4096_7168,
                stages_7168_2048=stages_7168_2048,
                stages_7168_256=stages_7168_256,
                pdl_level=pdl_level,
                fuse_swiglu=fuse_swiglu,
                SwiGLUOutputT=SwiGLUOutputT,
                swiglu_match_bf16=swiglu_match_bf16,
                swiglu_disable_compute=swiglu_disable_compute,
                swiglu_enable_trace=swiglu_enable_trace,
                TraceBufT=TraceBufT,
                swiglu_use_inplace=swiglu_use_inplace,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                num_active_experts,
                ctx,
                swiglu_out,
                trace_buf,
            )

        # Kimi K2.5 TP=8 up-proj: (N=512, K=7168) has an unusually small N
        # dimension, so the global-ablation winner for both (mma_bn, cta_group)
        # and stages differs from the regime-default classifier. All three
        # regimes converge on cta_group=2, stages=6; only mma_bn changes with
        # decode vs prefill. This path goes straight to
        # `_launch_grouped_block_scaled` with an explicit stages=6 (NOT via
        # `_regime`/`_dispatch_regime`, whose (N, K) stage table has no (512,
        # 7168) row and would fall through to stages=auto).
        @always_inline
        @__parameter
        def _launch512[mma_bn: Int, cta_group: Int]() raises:
            _launch_grouped_block_scaled[
                transpose_b,
                True,
                mma_bn,
                cta_group,
                num_pipeline_stages=6,
                pdl_level=pdl_level,
                fuse_swiglu=fuse_swiglu,
                SwiGLUOutputT=SwiGLUOutputT,
                swiglu_match_bf16=swiglu_match_bf16,
                swiglu_disable_compute=swiglu_disable_compute,
                swiglu_enable_trace=swiglu_enable_trace,
                TraceBufT=TraceBufT,
                swiglu_use_inplace=swiglu_use_inplace,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                a_offsets,
                a_scale_offsets,
                expert_ids,
                expert_scales,
                num_active_experts,
                ctx,
                swiglu_out,
                trace_buf,
            )

        comptime if N == 512 and K == 7168:
            if estimated_total_m <= num_active_experts * DECODE_AVG_M:
                _launch512[64, 2]()
            else:
                _launch512[128, 2]()
        else:
            # Three-regime classifier on avg_m = estimated_total_m /
            # num_active_experts, iterated as an ordered (upper-bound,
            # config) table. First-match-and-return reproduces the original
            # `if avg <= D / elif avg <= S / else` cascade exactly: the
            # decode row (upper=DECODE_AVG_M) wins first, else small-prefill
            # (upper=SMALL_PREFILL_AVG_M), else the unbounded large-prefill
            # row. Per-(N, K) tuned stages travel in each row, identical to
            # the prior explicit arms.
            #
            # (N=7168, K=2048) down-proj decode (mma_bn=8, cta_group=1):
            # B200 ablation (bench_grouped_matmul) shows stages 4->6 is a
            # no-regret win that grows with the active expert count: ~0%
            # at 8 active experts (grid too small to benefit), +11% at 12,
            # +5% at 16. The down-proj has only 8 K-iters (K=2048), so the
            # deeper pipeline overlaps cold-weight loads under more
            # concurrent CTAs as the grid widens; up-proj (s_4096_7168) is
            # already optimal at 6.
            comptime regimes = [
                # (upper_avg_m, mma_bn, cta_group, s_4096_7168, s_7168_2048,
                #  s_7168_256). upper_avg_m < 0 = unbounded (large prefill).
                (DECODE_AVG_M, 8, 1, 6, 6, -1),
                (SMALL_PREFILL_AVG_M, 64, 2, 6, 6, -1),
                (-1, 128, 2, 7, 6, 6),
            ]
            comptime for r in regimes:
                comptime _s0 = Optional[Int](r[3]) if r[3] >= 0 else Optional[
                    Int
                ](None)
                comptime _s1 = Optional[Int](r[4]) if r[4] >= 0 else Optional[
                    Int
                ](None)
                comptime _s2 = Optional[Int](r[5]) if r[5] >= 0 else Optional[
                    Int
                ](None)
                if r[0] < 0 or estimated_total_m <= num_active_experts * r[0]:
                    _regime[r[1], r[2], _s0, _s1, _s2]()
                    return


def grouped_matmul_mxfp8_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
    fuse_swiglu: Bool = False,
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput[
        MXFP8_SF_DTYPE, MXFP8_SF_VECTOR_SIZE
    ],
    swiglu_match_bf16: Bool = True,
    swiglu_disable_compute: Bool = False,
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    swiglu_use_inplace: Bool = False,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
    swiglu_out: SwiGLUOutputT = NullSwiGLUOutput[](),
    trace_buf: TraceBufT = NullTrace(),
) raises:
    """Dispatch grouped MXFP8 matmul with regime-classified configs.

    Mirror of `grouped_matmul_nvfp4_dispatch` for MXFP8 (E4M3 elements,
    E8M0 per-32-element scales). Hands off to the `KIND_MXF8F6F4`
    launcher with `AB_swapped=True` and the same avg-tokens-per-expert
    regime split as NVFP4:

        decode (avg_m <= 8):         mma_bn=8,   cta_group=1
        small prefill (avg_m <= 64): mma_bn=64,  cta_group=2
        large prefill:               mma_bn=128, cta_group=2

    Both the non-fused (BF16-output) and fused paths classify
    identically, so the fused-SwiGLU epilogue sees the same matmul
    output as the non-fused reference chain. Pipeline stages are
    auto-computed; per-(N, K) stage tuning like NVFP4's is a
    follow-up. Note: the dtype-routing dispatch
    `grouped_matmul_block_scaled_sm100_dispatch` still uses
    `AB_swapped=False, cta_group=1` for MXFP8; reconciling the two is
    a follow-up.

    When `fuse_swiglu=True`, the kernel emits packed MXFP8 plus a 5D
    E8M0 scale tile in place of the BF16 C store. Caller must
    pre-permute `W` on the N axis with `sigma(2i)=i, sigma(2i+1)=H+i`
    (same permutation pattern as the NVFP4 fused path).

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, MOGG interface compatibility).
        pdl_level: Programmatic dependent launch level.
        fuse_swiglu: When True, the kernel fuses SwiGLU + MXFP8 quant
            into the epilogue.
        SwiGLUOutputT: Trait impl for the fused-output sink. Defaults
            to the MXFP8-flavored `NullSwiGLUOutput` so the kernel ABI
            is byte-identical to the non-fused MXFP8 path when off.
        swiglu_match_bf16: When True, the SwiGLU+MXFP8 epilogue rounds
            through bf16 to match the chained reference's precision.
        swiglu_disable_compute: Diagnostic only.
        swiglu_enable_trace: When True, records per-tile timestamps.
        TraceBufT: Trait impl for the per-CTA timestamp buffer.
        swiglu_use_inplace: When True, the fused epilogue takes the
            in-place register path on the decode regime (honored only
            where mma_bn <= 8, same gating as NVFP4).

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K). MXFP8 (one fp8 per byte).
        b: Weight tensor B (num_experts, N, K). MXFP8.
        a_scales: Scale factors for A (5D, E8M0).
        b_scales: Scale factors for B (6D, E8M0).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
        swiglu_out: Sink carrier when `fuse_swiglu=True`. `NullSwiGLUOutput
            [MXFP8_SF_DTYPE, MXFP8_SF_VECTOR_SIZE]()` otherwise.
        trace_buf: Per-CTA timestamp buffer when trace is on.
    """
    # Per-(N, K) decode pipeline-stage override. The stage auto-maximizer
    # picks 12 stages for the (mma_bn=8, cta_group=1) decode tile, but this
    # thin, weight-load-BW-bound decode regime runs ~5-7% faster at 6 stages
    # (better weight-TMA cadence; numerically identical -- stages is pipeline
    # depth only). Scoped per-(N, K) so only the measured shapes change and
    # all others keep auto; decode regime only (the prefill rows below keep
    # the classifier's pick); non-fused only, matching what was measured.
    # Measured on B200 for an MXFP8 MoE thin-decode workload (EP8), stages=6
    # vs auto across M in {1, 8, 16, 32}:
    #   gate_up N=6144 K=6144: -5.3% (M16: 110.6 -> 104.7 us)
    #   down    N=6144 K=3072: -7.3% (M16:  59.8 ->  55.4 us)
    comptime _decode_N = type_of(c).static_shape[1]
    comptime _decode_K = type_of(a).static_shape[1]
    comptime _decode_stages = 6 if (
        not fuse_swiglu
        and _decode_N == 6144
        and (_decode_K == 6144 or _decode_K == 3072)
    ) else -1

    # Nested forwarder: the MXFP8 regimes differ in (mma_bn, cta_group) and
    # -- decode only -- pipeline stages; AB_swapped, scaling_kind, and all the
    # other forwarding comptime params + the 13 runtime args are identical.
    # Factor the call here and select via the same ordered (upper-bound,
    # config) table as the NVFP4 path. Stages travel per-row as an Int with
    # -1 = auto (the classifier's pick), mirroring the NVFP4 table sentinel.
    @always_inline
    @__parameter
    def _go[mma_bn: Int, cta_group: Int, stages: Optional[Int]]() raises:
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped=True,
            mma_bn=mma_bn,
            cta_group=cta_group,
            num_pipeline_stages=stages,
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
            pdl_level=pdl_level,
            fuse_swiglu=fuse_swiglu,
            SwiGLUOutputT=SwiGLUOutputT,
            swiglu_match_bf16=swiglu_match_bf16,
            swiglu_disable_compute=swiglu_disable_compute,
            swiglu_enable_trace=swiglu_enable_trace,
            TraceBufT=TraceBufT,
            swiglu_use_inplace=swiglu_use_inplace,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
            swiglu_out,
            trace_buf,
        )

    # (upper_avg_m, mma_bn, cta_group, stages_int). upper_avg_m < 0 =
    # unbounded (large prefill). Only the decode row carries a stage override;
    # prefill rows use -1 (= auto). First-match-and-return reproduces the
    # original decode / small-prefill / large-prefill cascade exactly.
    comptime regimes = [
        (DECODE_AVG_M, 8, 1, _decode_stages),
        (SMALL_PREFILL_AVG_M, 64, 2, -1),
        (-1, 128, 2, -1),
    ]
    comptime for r in regimes:
        comptime _s = Optional[Int](r[3]) if r[3] >= 0 else Optional[Int](None)
        if r[0] < 0 or estimated_total_m <= num_active_experts * r[0]:
            _go[r[1], r[2], _s]()
            return


def grouped_matmul_block_scaled_sm100_dispatch[
    transpose_b: Bool = True,
    target: StaticString = "cpu",
    pdl_level: PDLLevel = PDLLevel.ON,
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    estimated_total_m: Int,
    ctx: DeviceContext,
) raises:
    """Dispatch grouped block-scaled matmul based on input dtypes.

    Routes NVFP4 to shape-tuned decode/prefill dispatch, and MXFP4/MXFP8
    to a common launcher with default configs.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        target: Target device (unused, for MOGG interface compatibility).
        pdl_level: Programmatic dependent launch level.

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K//2 packed).
        b: Weight tensor B (num_experts, N, K//2 packed).
        a_scales: Scale factors for A (5D).
        b_scales: Scale factors for B (6D).
        a_offsets: Per-expert token offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        estimated_total_m: Estimated number of total non-padded tokens.
        ctx: Device context.
    """
    comptime scaling_kind = block_scaled_umma_kind[
        a.dtype, b.dtype, a_scales.dtype
    ]()

    # W4A8 lands in shared memory with the same byte geometry as MXFP8 -- the
    # FP4 TMA copy pads the weights on the way in -- so it takes the same
    # regime-tuned launcher.
    comptime if is_w4a8_operand_pair[a.dtype, b.dtype]():
        grouped_matmul_mxfp8_dispatch[transpose_b, target, pdl_level=pdl_level](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
        )
    elif scaling_kind == UMMAKind.KIND_MXF4NVF4:
        grouped_matmul_nvfp4_dispatch[transpose_b, target, pdl_level=pdl_level](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            estimated_total_m,
            ctx,
        )
    elif scaling_kind == UMMAKind.KIND_MXF4:
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped=False,
            mma_bn=128,
            cta_group=1,
            scaling_kind=UMMAKind.KIND_MXF4,
            pdl_level=pdl_level,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
    elif scaling_kind == UMMAKind.KIND_MXF8F6F4:
        _launch_grouped_block_scaled[
            transpose_b,
            AB_swapped=False,
            mma_bn=128,
            cta_group=1,
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
            pdl_level=pdl_level,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            a_offsets,
            a_scale_offsets,
            expert_ids,
            expert_scales,
            num_active_experts,
            ctx,
        )
