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
"""4-wave FP8 implicit-GEMM convolution for AMD MI355X (CDNA4).

Host launcher: `amd_4wave_conv()`

The kernel body lives on `AMD4WaveMatmul.run_conv2d` (in
`linalg.matmul.gpu.amd.amd_4wave_matmul`); the conv2d and matmul
share the same struct, the same 4-warp 2x2 quadrant layout, the same
MFMA shapes, and the same software-pipeline schedule. They differ
only in the A-operand loader: matmul uses `TileLoaderLDS` (linear
`[M, K]`); conv uses `TileLoaderLDSIm2col` (NHWC input + in-line
im2col address math). This file just bundles the launcher and the
HK chiplet/L2-swizzle helper.

Supported configuration space:

  - Filter R × S: any R, S >= 1.
  - Stride: any >= 1.
  - Dilation: any >= 1.
  - Pad: any >= 0 (halo lanes route to the SRD-OOB sentinel).
  - Input dtype: FP8 (E4M3FN), BF16, or FP16. Output dtype is BF16
    (FP8 in, BF16 out) or matches the input for BF16/FP16. All
    dtypes route through the framework-scheduled body.
  - C_in: any positive value. The loader picks between a fast-path
    "uniform substrip per call" code when `C_in % BK == 0` and
    `BK <= C_in`, and a slower per-lane-substrip code otherwise
    (e.g. ResNet stem with C_in = 64 and BK = 128).
  - K = R*S*C_in: must be a multiple of 2*BK = 256 (the 4-wave
    schedule's two-stage prologue requirement). When R*S*C_in isn't
    aligned, the caller K-pads the filter buffer to a multiple of
    2*BK by zero-filling the trailing K rows; pass the real C_in via
    the `C_in` comptime kwarg on the launcher so the loader uses the
    unpadded value in its address math. Zero filter rows make the
    MMA contribution for padded K columns 0 regardless of what the
    A loader produces.
  - num_splits: 1 (no split-K).
  - BM × BN: 64×64, 128×128, or 128×256 (from the matmul's auto-pick;
    overridable via `block_m_override` / `block_n_override`).

The conv launcher takes the 4D NHWC input directly. The filter is a
2D `[Cout, K_padded]` tile-tensor in FRSC order (filter row `f`
column `k = r*S*C_in + s*C_in + c` is `weight[f, r, s, c]`). The
output is a 2D `[N*H_out*W_out, Cout]` view of the NHWC output
buffer: for packed NHWC layouts the 2D view aliases the same bytes
exactly. See `max/kernels/test/gpu/nn/test_amd_4wave_conv*.mojo` for
end-to-end correctness coverage.
"""

from std.math import ceildiv
from std.utils import Index

from max.gpu.host import DeviceContext

from layout import TileTensor

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from linalg.matmul.gpu.amd.amd_4wave_matmul import (
    AMD4WaveMatmul,
    Conv2DKernelConfig,
    MatmulKernelConfig,
)


# ===----------------------------------------------------------------------=== #
# HK chiplet / L2 swizzle (XCD-stride remap + WGM=4 column-major grouping)
# ===----------------------------------------------------------------------=== #


@always_inline
def _xcd_wgm_swizzle(
    wgid_raw: Int, num_pid_m: Int, num_pid_n: Int
) -> Tuple[Int, Int]:
    """HipKittens chiplet + L2 swizzle for MI355X (CDNA4).

    Stage 1 (XCD remap): MI355X has 8 chiplets (XCDs) with their own
    L2 slices. Raw block_idx maps adjacent wgids to adjacent CUs which
    typically share an XCD; this remap spreads adjacent wgids across
    XCDs.

    Stage 2 (WGM column-major group): walks WGM=4 row-blocks together
    before advancing N. Improves L2 reuse on the shared operand when
    each CU processes multiple WGs of a row-group.

    Mirrors `4_wave.cu` lines 117-136 / 353-372 in HipKittens.

    The swizzle's L2 reuse only pays off when the WG count is large
    enough that each CU runs multiple WGs in one kernel launch, i.e.
    `num_wgs > num_CUs`. For our decode/prefill shapes (typically
    32-1024 WGs vs 304 CUs), the swizzle math would just add overhead
    without any L2 benefit. We gate on `num_wgs > 4 * num_CUs` to
    skip the swizzle on those shapes; only at the largest shapes
    (e.g. 16k³ at BM=BN=128 = 16384 WGs) does it actually help.
    """
    comptime NUM_XCDS = 8
    comptime WGM = 4
    comptime NUM_CUS = 304  # MI355X
    comptime SWIZZLE_THRESHOLD = 4 * NUM_CUS

    var num_wgs = num_pid_m * num_pid_n

    # Trivial row-major decode for small grids: skip the XCD remap +
    # WGM grouping (one divmod here, three on the full path).
    if num_wgs <= SWIZZLE_THRESHOLD or num_wgs % NUM_XCDS != 0:
        var pid_m, pid_n = divmod(wgid_raw, num_pid_n)
        return (pid_m, pid_n)

    # Full HK swizzle: XCD remap + WGM=4 column-major group.
    # intra_xcd is the wg's position within the XCD's allocation
    # (consecutive wgs every NUM_XCDS apart); xcd is which XCD the wg
    # lands on (adjacent raw wgids spread across XCDs).
    var intra_xcd, xcd = divmod(wgid_raw, NUM_XCDS)
    var wgid = xcd * (num_wgs // NUM_XCDS) + intra_xcd
    var num_wgid_in_group = WGM * num_pid_n
    var group_id, intra_group = divmod(wgid, num_wgid_in_group)
    var first_pid_m = group_id * WGM
    var group_size_m = min(num_pid_m - first_pid_m, WGM)
    var pid_n, intra_group_m = divmod(intra_group, group_size_m)
    var pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


@always_inline
def amd_4wave_conv[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    enable_swizzle: Bool = True,
    block_m_override: Int = 0,
    block_n_override: Int = 0,
    block_k_override: Int = 0,
    dump_asm_path: StaticString = "",
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    # Pre-residual fused compute lambda. Fires on the post-cast
    # `c_type` MMA output BEFORE the residual FMA. Use for bias /
    # ReLU / SiLU / GELU fusion. Matches the SM100 reference
    # `D = lambda(Conv(A,B)) + beta * C` semantics. Signature
    # `(IndexList[2], SIMD) capturing -> SIMD`.
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    # Opt-in in-kernel residual: when True, the kernel bulk-prefetches
    # `source_ptr[m, n]` into VGPRs at the start of the epilogue and
    # applies `out = mma + beta * source` before the store. Replaces
    # the slower lambda-fusion residual (~24% faster on memory-bound
    # shapes — see `amd_4wave_conv_fprop_with_residual`). When False
    # (default), the kernel is identical to the no-residual variant
    # and the `source_ptr`/`source_row_stride`/`beta` args are unused
    # at comptime.
    has_residual: Bool = False,
    # Conv geometry (defaults reproduce the M1 1×1 case).
    H: Int = 1,
    W: Int = 1,
    H_out: Int = 1,
    W_out: Int = 1,
    R: Int = 1,
    S: Int = 1,
    stride_h: Int = 1,
    stride_w: Int = 1,
    dilation_h: Int = 1,
    dilation_w: Int = 1,
    pad_h: Int = 0,
    pad_w: Int = 0,
    # Real input channel count. 0 (default) means infer from filter
    # (no K-padding). Set explicitly when caller has K-padded the
    # filter for shapes where Q*R*S*C_in isn't a multiple of 2*BK = 256.
    C_in: Int = 0,
    # Runtime-HW path: when True, H/W/H_out/W_out (and D/D_out in 3D
    # mode) above are ignored; the kernel reads input.dim[1] /
    # input.dim[2] (and input.dim[3] for 3D) at runtime and recomputes
    # the output spatial dims from `(R, S, stride, dilation, pad)` (and
    # `(Q, stride_d, dilation_d, pad_d)` for 3D). Use this for
    # graph-compiled callers (e.g. FLUX VAE) with symbolic image
    # resolution. See `TileLoaderLDSIm2col.use_runtime_hw` for the perf
    # cost (one runtime divmod per `load_tile` on the BF16 path; one
    # per (lane, iter) on the FP8 swizzle path).
    use_runtime_hw: Bool = False,
    # 3D-conv mode (NDHWC + Q×R×S filter). Defaults reproduce 2D mode
    # so existing 2D callers compile unchanged. Q > 1 activates 3D
    # mode: the launcher expects a 5D NDHWC `a` tensor and the kernel
    # decomposes M = N*D_out*H_out*W_out, K = Q*R*S*C.
    Q: Int = 1,
    D: Int = 1,
    D_out: Int = 1,
    stride_d: Int = 1,
    dilation_d: Int = 1,
    pad_d: Int = 0,
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
    # Residual args. Only used when `has_residual=True`. The defaults
    # (dangling placeholder pointer, 0 stride, 0.0 beta) are picked up
    # by the kernel only when has_residual=False — the args are then
    # dead-code-eliminated by the compiler (still in the launch packet,
    # 16 bytes overhead per launch).
    source_ptr: UnsafePointer[Scalar[c_type], ImmutAnyOrigin] = UnsafePointer[
        Scalar[c_type], ImmutAnyOrigin
    ].unsafe_dangling(),
    source_row_stride: Int = 0,
    beta: Float32 = 0.0,
) raises:
    """Launches the 4-wave implicit-GEMM convolution on the device.

    The caller passes the 4D NHWC input directly; the output is a 2D
    `[N*H_out*W_out, Cout]` view of the NHWC output buffer (caller does
    the reshape: for packed NHWC, the 2D view aliases the same memory
    exactly). Defaults reproduce the 1×1 pointwise case where the address
    math collapses to `m*C + k`.

    Dtype-dependent MFMA shape (selected at comptime); all dtypes
    route through the framework-scheduled body:

      - FP8 (E4M3FN): MMA shape `(16, 16, 128)`, BK=128.
      - BF16 / FP16: MMA shape `(16, 16, 32)`, BK=128 by default
        (override via `block_k_override`).

    Parameters:
        a_type: Input element type. Must equal `b_type`. One of
            `float8_e4m3fn`, `bfloat16`, `float16`.
        b_type: Filter element type.
        c_type: Output element type.
        enable_swizzle: Enable LDS bank-conflict avoidance.
        block_m_override: If > 0, force BM (must be 64 or 128).
        block_n_override: If > 0, force BN (must be 64, 128, or 256).
        block_k_override: If > 0, force BK. Must be a multiple of
            `MMA_K` (32 for bf16/fp16, 128 for FP8) and in
            {32, 64, 128}. Default 0 → auto-pick.
        dump_asm_path: If non-empty, dump compiled GCN assembly to path.
        elementwise_lambda_fn: Optional fused epilogue lambda.
        elementwise_compute_lambda_fn: Optional pre-residual fused
            compute lambda. Fires on the post-cast `c_type` MMA output
            BEFORE the residual FMA, matching the SM100
            `D = lambda(Conv(A,B)) + beta * C` ordering. Use for
            bias / ReLU / SiLU / GELU fusion.
        has_residual: When True, the kernel bulk-prefetches
            `source_ptr[m, n]` into VGPRs at the start of the epilogue
            and applies `out = mma + beta * source` before the store.
            When False (default), `source_ptr` / `source_row_stride` /
            `beta` are unused and the epilogue is byte-identical to the
            no-residual kernel.
        H: Input height.
        W: Input width.
        H_out: Output height (= `(H + 2*pad_h - dilation_h*(R-1) - 1) //
            stride_h + 1`).
        W_out: Output width.
        R: Filter height.
        S: Filter width.
        stride_h: Vertical conv stride (>= 1).
        stride_w: Horizontal conv stride (>= 1).
        dilation_h: Vertical dilation (>= 1).
        dilation_w: Horizontal dilation (>= 1).
        pad_h: Vertical pad (>= 0). Halo lanes route to the SRD-OOB
            sentinel so the read returns 0.
        pad_w: Horizontal pad (>= 0).
        C_in: Real input channel count when the caller has K-padded the
            filter for shapes where R*S*C_in is not a multiple of
            2*BK = 256. Default 0 means "infer from K_filter // (R*S)"
            (no K-padding).
        use_runtime_hw: When True, H/W/H_out/W_out (and D/D_out in 3D
            mode) are read from `a.dim()` at runtime instead of the
            comptime params above. Required for graph-compiled callers
            with symbolic image resolution (e.g. FLUX VAE). The
            K-decomposition and conv params (Q, R, S, stride, dilation,
            pad, C_in) still need to be static.
        Q: Filter temporal extent. `Q == 1` (default) keeps the kernel
            in 2D mode (4D NHWC input). `Q > 1` activates 3D mode (5D
            NDHWC input, K = Q*R*S*C); the loader decomposes
            `M = N*D_out*H_out*W_out` and `K = Q*R*S*C` internally.
        D: Input temporal depth (3D mode only; unused when Q == 1).
        D_out: Output temporal depth (3D mode only).
        stride_d: Conv temporal stride (3D mode only).
        dilation_d: Conv temporal dilation (3D mode only).
        pad_d: Temporal pad (3D mode only). Halo lanes route to the
            SRD-OOB sentinel for zero-clamp behavior.

    Args:
        a: 4D NHWC input tile-tensor of shape `(N, H, W, C)`.
        b: 2D filter tile-tensor of shape `(Cout, K)` where K = R*S*C_in
            (possibly K-padded to a multiple of 256).
        c: 2D output tile-tensor of shape `(N*H_out*W_out, Cout)`:
            a view of the NHWC output buffer.
        ctx: Device context used to enqueue the kernel.
        source_ptr: Pointer to the residual buffer (NHWC-contiguous,
            same shape as output). Only read when `has_residual=True`;
            callers leave the default (null) when `has_residual=False`.
        source_row_stride: Element stride of the residual's row
            dimension (`C_out` for NHWC contiguous). Unused when
            `has_residual=False`.
        beta: Residual scale factor (`D = Conv + beta * source`).
            Unused when `has_residual=False`.

    Raises:
        An error if device enqueue fails.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type.is_float8() or a_type == .bfloat16 or a_type == .float16
    ), "4-wave conv supports float8_e4m3fn, bfloat16, or float16"

    # MMA K-dim selection: FP8 uses MFMA 16x16x128; bf16/fp16 use MFMA
    # 16x16x32. `num_k_mmas = BK/MMA_K` MFMAs fire along K internally
    # via `QuadrantMmaOp`, so BK > MMA_K is transparent to the body.
    comptime _is_fp8 = a_type.is_float8()
    comptime _mma_k = 128 if _is_fp8 else 32
    comptime _bk = block_k_override if block_k_override > 0 else 128
    comptime assert block_k_override == 0 or (
        block_k_override % _mma_k == 0 and block_k_override in (32, 64, 128)
    ), (
        "block_k_override must be 0 (auto) or a multiple of MMA_K in"
        " {32, 64, 128}"
    )

    var N = Int(c.dim[1]())
    var M = Int(c.dim[0]())

    # Bundle the conv geometry once — passed to `run_conv2d` below.
    # Q > 1 activates 3D mode inside the kernel (the loader expects
    # a 5D NDHWC `a`); Q=1 defaults to 2D mode and the 3D-only fields
    # (D, D_out, stride_d, dilation_d, pad_d) are ignored.
    comptime conv_cfg = Conv2DKernelConfig(
        H=H,
        W=W,
        H_out=H_out,
        W_out=W_out,
        R=R,
        S=S,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        pad_h=pad_h,
        pad_w=pad_w,
        C_in=C_in,
        use_runtime_hw=use_runtime_hw,
        Q=Q,
        D=D,
        D_out=D_out,
        stride_d=stride_d,
        dilation_d=dilation_d,
        pad_d=pad_d,
    )

    @always_inline
    @__parameter
    def run_kernel[config: MatmulKernelConfig]() raises:
        # Dispatch via `AMD4WaveMatmul.run_conv2d` — the unified entry
        # point that hosts the 4-wave conv2d body alongside the matmul.
        # `has_residual` carries through to the kernel's epilogue;
        # source/stride/beta become unused-but-present runtime args
        # when has_residual=False (DCE'd, 16 bytes launch packet cost).
        comptime kernel = AMD4WaveMatmul[
            a_type,
            b_type,
            c_type,
            config,
            enable_swizzle,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ].run_conv2d[
            conv_cfg,
            a.LayoutType,
            b.LayoutType,
            c.LayoutType,
            has_residual=has_residual,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ]

        # 1D launch grid for the HK chiplet/L2 swizzle. The kernel
        # decodes (pid_m, pid_n) from block_idx.x.
        var num_blocks_n = ceildiv(N, config.block_shape[1])
        var num_blocks_m = ceildiv(M, config.block_shape[0])
        comptime if dump_asm_path != "":
            ctx.enqueue_function[kernel, dump_asm=dump_asm_path](
                a,
                b,
                c,
                source_ptr,
                Int32(source_row_stride),
                beta,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )
        else:
            ctx.enqueue_function[kernel](
                a,
                b,
                c,
                source_ptr,
                Int32(source_row_stride),
                beta,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )

    # Tile shape: BK depends on dtype default + `block_k_override`.
    # MMA K-dim: 128 for FP8, 32 for bf16/fp16. 4 warps in 2x2 grid:
    #   BM in {64, 128} (smaller fails QuadrantMmaOp's even-mma assert)
    #   BN in {64, 128, 256} (smaller fails same)

    comptime if block_m_override > 0 and block_n_override > 0:
        # Caller-specified config. The kernel's directional asymmetric
        # bug: BN < BM produces wrong results (~33% of elements off);
        # BN >= BM is correct. Until the kernel-level assumption is
        # fixed, we require BN >= BM.
        comptime BM_o = block_m_override
        comptime BN_o = block_n_override
        comptime assert (
            BM_o == 64 or BM_o == 128
        ), "block_m_override must be 64 or 128"
        comptime assert (
            BN_o == 64 or BN_o == 128 or BN_o == 256
        ), "block_n_override must be 64, 128, or 256"
        comptime assert (
            BN_o >= BM_o
        ), "block_n_override must be >= block_m_override (BN < BM is broken)"
        comptime config_override = MatmulKernelConfig(
            block_shape=Index(BM_o, BN_o, _bk),
            warp_shape=Index(BM_o // 2, BN_o // 2, _bk),
            mma_shape=Index(16, 16, _mma_k),
        )
        run_kernel[config_override]()
        return

    # Auto-pick. FP8 heuristic from the symmetric (BN==BM) matmul sweep
    # at N=K=4096 (M ≤ 512 → BM=64; M > 512 → BM=128). bf16/fp16
    # default to BM=BN=128 (matching `structured_4wave_matmul`'s
    # default for the prefill regime — production dispatchers should
    # pin tile shape via the overrides).
    comptime config_64 = MatmulKernelConfig(
        block_shape=Index(64, 64, _bk),
        warp_shape=Index(32, 32, _bk),
        mma_shape=Index(16, 16, _mma_k),
    )
    comptime config_128 = MatmulKernelConfig(
        block_shape=Index(128, 128, _bk),
        warp_shape=Index(64, 64, _bk),
        mma_shape=Index(16, 16, _mma_k),
    )

    comptime if _is_fp8:
        if M <= 512:
            run_kernel[config_64]()
        else:
            run_kernel[config_128]()
    else:
        run_kernel[config_128]()
