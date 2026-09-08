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
"""4-wave matmul for AMD MI355X (CDNA4).

Entry point: `AMD4WaveMatmul.run()` (matmul) and `.run_conv2d()` (conv).
Host launcher: `structured_4wave_matmul()` (matmul-only); the conv
launcher lives in `nn/conv/gpu/amd/amd_4wave_conv.mojo`.

4-warp 2x2 quadrant layout with cross-stage register rotation,
adapted from HipKittens FP8_4wave's `matmul_device_*`:

  - 4 mini-iters per loop iter, each with `G_load + frag_load + mma_ABt`.
  - Cross-stage register rotation: a[0]/b[0] are reloaded mid-iter from
    the `next` stage so iter k+1's first MMA can fire without waiting
    on LDS.
  - 2-iter epilogue drain.

Body is driven by the framework schedule (`Pipeline4Wave` in
`amd_4wave_schedule.mojo`) under `SchedulingStrategy.IDENTITY` +
`minimal_barriers` + `omit_mma_set_prio`. The schedule consumes the
logical 24-op cross-stage-rotation body and derives wait counts /
barriers from `KernelGeometry`. Supports FP8 (E4M3FN), BF16, and FP16
through a single body: MMA shape and BK select on dtype.
"""

from std.bit import log2_floor
from std.math import ceildiv
from std.sys import align_of, size_of, llvm_intrinsic
from std.sys.intrinsics import readfirstlane
from std.utils import Index, IndexList, StaticTuple
from std.collections import Array
from std.utils.numerics import get_accum_type

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from max.gpu.intrinsics import AMDBufferResource
from max.gpu.sync import schedule_barrier, s_waitcnt

from layout import TensorLayout, TensorEngine, TileTensor
from layout.swizzle import Swizzle
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation

from pipeline.config import ScheduleConfig, SchedulingStrategy
from pipeline.geometry import KernelGeometry
from pipeline.pipeline_dsl import ScheduleEntry
from pipeline.program_builder import derive_safe_max_globals
from pipeline.types import _Ops

from structured_kernels.amd_tile_io import RegTileEpilogue, TileLoaderLDS
from structured_kernels.amd_tile_io_conv import TileLoaderLDSIm2col

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from .amd_target import mi355x_target
from .amd_4wave_schedule import (
    LOAD_A,
    LOAD_B,
    MMA_LOAD_A,
    MMA_LOAD_B,
    MMA,
    build_schedule,
)
from .matmul_mma import QuadrantMmaOp


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
    enough that each CU runs multiple WGs in one kernel launch: i.e.
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


# ===----------------------------------------------------------------------=== #
# MatmulKernelConfig / Conv2DKernelConfig
# ===----------------------------------------------------------------------=== #


struct MatmulKernelConfig(ImplicitlyCopyable, Movable, Writable):
    """Block/warp/MMA shape configuration for 4-wave kernels.

    Shared by `AMD4WaveMatmul`'s matmul and conv2d entry points: both
    use the same 4-warp 2×2 quadrant layout and the same MFMA shape
    selection, so the per-call surface only needs the workgroup /
    warp / MMA tile shapes.
    """

    var block_shape: IndexList[3]
    """Workgroup tile shape `(BM, BN, BK)`."""
    var warp_shape: IndexList[3]
    """Per-warp tile shape `(WM, WN, WK)`."""
    var mma_shape: IndexList[3]
    """Single-MMA instruction shape `(MMA_M, MMA_N, MMA_K)`."""

    def __init__(
        out self,
        *,
        block_shape: IndexList[3],
        warp_shape: IndexList[3],
        mma_shape: IndexList[3],
    ):
        """Constructs a `MatmulKernelConfig` from the three tile shapes.

        Args:
            block_shape: Workgroup tile shape `(BM, BN, BK)`.
            warp_shape: Per-warp tile shape `(WM, WN, WK)`.
            mma_shape: Single-MMA instruction shape `(MMA_M, MMA_N, MMA_K)`.
        """
        self.block_shape = block_shape
        self.warp_shape = warp_shape
        self.mma_shape = mma_shape

    @staticmethod
    def _write_index_list(
        mut writer: Some[Writer], list: IndexList, sep: StaticString
    ):
        comptime for i in range(list.size):
            if i != 0:
                writer.write(sep)
            writer.write(list[i])

    @always_inline
    def num_threads(self) -> Int:
        """Returns the total threads per workgroup (warps x `WARP_SIZE`).

        Returns:
            The number of threads in one workgroup.
        """
        var num_warps = self.block_shape // self.warp_shape
        return num_warps.flattened_length() * WARP_SIZE

    def write_to(self, mut writer: Some[Writer]):
        """Writes a human-readable shape tag to `writer`.

        Args:
            writer: Sink for the rendered config tag.
        """
        writer.write("4wave_config_")
        Self._write_index_list(writer, self.block_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.warp_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.mma_shape, "x")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this config to `writer`.

        Args:
            writer: Sink for the rendered config tag.
        """
        self.write_to(writer)


# Back-compat alias for older sources/tests that imported the legacy
# name (e.g. `test_amd_matmul_asm.mojo`). New code should prefer
# `MatmulKernelConfig`.
comptime KernelConfig = MatmulKernelConfig


struct Conv2DKernelConfig(ImplicitlyCopyable, Movable, Writable):
    """Conv-specific geometry for `AMD4WaveMatmul`'s conv2d entry point.

    Companion to `MatmulKernelConfig`: the matmul shape config drives
    the 4-wave block/warp/MMA tiling, while this struct carries the
    extra conv2d parameters the im2col loader needs to materialize the
    A operand from a 4D NHWC input. Pass both at the kernel callsite.
    """

    var H: Int
    """Input spatial height (NHWC dim 1). Used as `H_eff` when
    `use_runtime_hw=False`; ignored otherwise (loader reads runtime H)."""
    var W: Int
    """Input spatial width (NHWC dim 2). Same semantics as `H`."""
    var H_out: Int
    """Output spatial height = `(H + 2*pad_h - dilation_h*(R-1) - 1)
    // stride_h + 1`. Comptime-only on the static-HW path."""
    var W_out: Int
    """Output spatial width = `(W + 2*pad_w - dilation_w*(S-1) - 1)
    // stride_w + 1`."""
    var R: Int
    """Filter spatial height."""
    var S: Int
    """Filter spatial width."""
    var stride_h: Int
    """Conv vertical stride (>= 1)."""
    var stride_w: Int
    """Conv horizontal stride (>= 1)."""
    var dilation_h: Int
    """Conv vertical dilation (>= 1)."""
    var dilation_w: Int
    """Conv horizontal dilation (>= 1)."""
    var pad_h: Int
    """Vertical pad (>= 0). When > 0, halo lanes route to the SRD-OOB
    sentinel for zero-clamp behavior."""
    var pad_w: Int
    """Horizontal pad (>= 0)."""
    var C_in: Int
    """Real input channel count. When > 0, lets the caller K-pad the
    filter (allocate filter as `[Cout, K_padded]` where K_padded =
    round_up(R*S*C_in, 2*BK) and zero-fill the trailing K rows). When
    0, the kernel derives `C_in = K_filter // (R*S)` and asserts
    exact divisibility. In 3D mode the divisor is `Q*R*S`."""
    var use_runtime_hw: Bool
    """When True, H/W/H_out/W_out (and D/D_out in 3D mode) are runtime
    values read from the input tensor (typically via graph-compiler
    symbolic resolution). When False, all conv geometry is comptime."""
    var Q: Int
    """Filter temporal extent (3D only). `Q == 1` keeps the kernel in
    2D mode (4D NHWC input, K = R*S*C). `Q > 1` activates 3D mode (5D
    NDHWC input, K = Q*R*S*C)."""
    var D: Int
    """Input temporal depth (NDHWC dim 1). 2D mode: unused."""
    var D_out: Int
    """Output temporal depth. 2D mode: unused."""
    var stride_d: Int
    """Conv temporal stride (>= 1). 2D mode: unused."""
    var dilation_d: Int
    """Conv temporal dilation (>= 1). 2D mode: unused."""
    var pad_d: Int
    """Temporal pad (>= 0). 2D mode: unused. When > 0, halo lanes route
    to the SRD-OOB sentinel for zero-clamp behavior."""

    def __init__(
        out self,
        *,
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
        C_in: Int = 0,
        use_runtime_hw: Bool = False,
        # 3D-only (default to 2D-equivalent so 2D callers don't have to
        # set them). Q > 1 activates 3D mode.
        Q: Int = 1,
        D: Int = 1,
        D_out: Int = 1,
        stride_d: Int = 1,
        dilation_d: Int = 1,
        pad_d: Int = 0,
    ):
        """Constructs a `Conv2DKernelConfig` from the conv geometry.

        Args:
            H: Input spatial height.
            W: Input spatial width.
            H_out: Output spatial height.
            W_out: Output spatial width.
            R: Filter spatial height.
            S: Filter spatial width.
            stride_h: Conv vertical stride.
            stride_w: Conv horizontal stride.
            dilation_h: Conv vertical dilation.
            dilation_w: Conv horizontal dilation.
            pad_h: Vertical pad.
            pad_w: Horizontal pad.
            C_in: Real input channel count.
            use_runtime_hw: When True, treat H/W/H_out/W_out (and
                D/D_out in 3D mode) as placeholders to be replaced
                at kernel-launch time with values read from the input
                tensor.
            Q: Filter temporal extent. `Q == 1` (default) keeps the
                kernel in 2D mode; `Q > 1` activates 3D mode.
            D: Input temporal depth (3D mode).
            D_out: Output temporal depth (3D mode).
            stride_d: Conv temporal stride (3D mode).
            dilation_d: Conv temporal dilation (3D mode).
            pad_d: Temporal pad (3D mode).
        """
        self.H = H
        self.W = W
        self.H_out = H_out
        self.W_out = W_out
        self.R = R
        self.S = S
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.C_in = C_in
        self.use_runtime_hw = use_runtime_hw
        self.Q = Q
        self.D = D
        self.D_out = D_out
        self.stride_d = stride_d
        self.dilation_d = dilation_d
        self.pad_d = pad_d

    def write_to(self, mut writer: Some[Writer]):
        """Writes a compact conv geometry tag to `writer`.

        Args:
            writer: Sink for the rendered tag.
        """
        writer.write("conv_")
        writer.write(self.H)
        writer.write("x")
        writer.write(self.W)
        writer.write("_R")
        writer.write(self.R)
        writer.write("S")
        writer.write(self.S)
        writer.write("_s")
        writer.write(self.stride_h)
        writer.write("x")
        writer.write(self.stride_w)
        writer.write("_p")
        writer.write(self.pad_h)
        writer.write("x")
        writer.write(self.pad_w)
        if self.dilation_h != 1 or self.dilation_w != 1:
            writer.write("_d")
            writer.write(self.dilation_h)
            writer.write("x")
            writer.write(self.dilation_w)
        if self.use_runtime_hw:
            writer.write("_rtHW")

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of this conv config.

        Args:
            writer: Sink for the rendered tag.
        """
        self.write_to(writer)


struct AMD4WaveMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    config: MatmulKernelConfig,
    /,
    enable_swizzle: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Hand-written 4-warp 2x2 inline-MMA matmul for AMD MI355X.

    Line-by-line port of HipKittens FP8_4wave's `matmul_device_1024`
    (BM=64) and `matmul_device_2048` (BM=128). No declarative pipeline
    framework: explicit waits, explicit barriers, explicit register
    rotation matching the source's k+2 prefetch pattern.

    Parameters:
        a_type: Input A element type.
        b_type: Input B element type.
        c_type: Output C element type.
        config: MatmulKernelConfig with block/warp/mma shapes.
        enable_swizzle: Enable LDS bank conflict avoidance.
        elementwise_lambda_fn: Optional epilogue.
    """

    comptime BM = Self.config.block_shape[0]
    """Workgroup M-tile size."""
    comptime BN = Self.config.block_shape[1]
    """Workgroup N-tile size."""
    comptime BK = Self.config.block_shape[2]
    """Workgroup K-tile size."""

    comptime WM = Self.config.warp_shape[0]
    """Per-warp M-tile size."""
    comptime WN = Self.config.warp_shape[1]
    """Per-warp N-tile size."""

    comptime MMA_M = Self.config.mma_shape[0]
    """Single-MMA M dimension."""
    comptime MMA_N = Self.config.mma_shape[1]
    """Single-MMA N dimension."""
    comptime MMA_K = Self.config.mma_shape[2]
    """Single-MMA K dimension."""

    comptime num_warps_m = Self.BM // Self.WM
    """Number of warps in the M dimension of the workgroup grid."""
    comptime num_warps_n = Self.BN // Self.WN
    """Number of warps in the N dimension of the workgroup grid."""
    comptime total_warps = Self.num_warps_m * Self.num_warps_n
    """Total warps per workgroup (must be 4 for this kernel)."""

    comptime num_m_mmas = Self.WM // Self.MMA_M
    """Number of MMAs per warp in the M dimension."""
    comptime num_n_mmas = Self.WN // Self.MMA_N
    """Number of MMAs per warp in the N dimension."""
    comptime num_k_mmas = Self.BK // Self.MMA_K
    """Number of MMAs per warp in the K dimension."""

    comptime in_type = Self.a_type
    """Input element type (A and B share a type for FP8)."""

    comptime simd_width = 16 // size_of[Self.in_type]()
    """SIMD lane width matched to AMD `buffer_load_lds`'s 16-byte transaction.

    Target-independent (unlike `simd_width_of[in_type]()`, which would
    return the host's SIMD width when this struct is instantiated from a
    comptime context running on the CPU host). `validate_config` asserts
    that `size_of[in_type]` divides 16 evenly; for any dtype that does
    (FP8/BF16/FP16/FP32) this gives the same value as `simd_width_of`
    inside an AMDGPU kernel.
    """
    comptime accum_dtype = get_accum_type[Self.c_type]()
    """Accumulator dtype derived from `c_type`."""

    comptime c_frag_size = Self.MMA_M * Self.MMA_N // WARP_SIZE
    """Per-lane output-fragment width for one MMA."""

    # Half-tile dimensions: each SMEM tile covers one M-subtile (or N-subtile)
    # of the WG-level BM x BK (or BN x BK) block.
    comptime half_BM = Self.BM // 2
    """Half of `BM`: one M-subtile per SMEM stage."""
    comptime half_BN = Self.BN // 2
    """Half of `BN`: one N-subtile per SMEM stage."""
    comptime mma_tile_m = Self.WM // 2
    """Per-quadrant M-tile size consumed by an MMA load."""
    comptime mma_tile_n = Self.WN // 2
    """Per-quadrant N-tile size consumed by an MMA load."""

    # Quadrant counts (pass-through to QuadrantMmaOp).
    comptime quadrant_m_mmas = Self.num_m_mmas // 2
    """Number of MMAs per quadrant in the M dimension."""
    comptime quadrant_n_mmas = Self.num_n_mmas // 2
    """Number of MMAs per quadrant in the N dimension."""

    # Producer thread layout for TileLoaderLDS.
    comptime _thread_cols = Self.BK // Self.simd_width

    # Swizzle geometry (shared between producer-side byte_swizzle and
    # consumer-side mma_swizzle). MI355X has 64 LDS banks x 4 bytes;
    # without swizzling the MMA thread access pattern causes 4-way bank
    # conflicts. FP8 16x16x128 uses a split-LDS path with 16-byte
    # fragments; everything else uses full MMA-fragment loads.
    #   FP8 16x16x128: log_tile=3, frag=16B -> Swizzle(3, 4, 4)
    comptime _elem_size = size_of[Self.in_type]()
    comptime _mma_frag_w = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime _use_split_lds = (
        Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime _lds_frag_w = 16 if Self._use_split_lds else Self._mma_frag_w
    comptime _swizzle_log_tile = log2_floor(Self.MMA_K // 32) + 1
    comptime _frag_bytes = Self._lds_frag_w * Self._elem_size
    comptime _swizzle_subtile_cols = 4 * Self.simd_width

    # mma_swizzle (consumer side, element-space): base is always
    # log2_floor(_frag_bytes) regardless of dtype.
    comptime mma_swizzle = Optional(
        Swizzle(Self._swizzle_log_tile, log2_floor(Self._frag_bytes), 4)
    ) if Self.enable_swizzle else Optional[Swizzle]()
    """Optional consumer-side MMA swizzle, populated when swizzling is on."""

    # byte_swizzle (producer side, byte-space, for load_to_lds writes):
    # FP8 base matches mma_swizzle; non-FP8 base uses subtile-col math.
    comptime _byte_swizzle_base = (
        log2_floor(Self._frag_bytes) if Self.in_type.is_float8() else (
            log2_floor(Self._swizzle_subtile_cols // 2)
            + log2_floor(Self._elem_size)
        )
    )
    comptime byte_swizzle = Optional(
        Swizzle(Self._swizzle_log_tile, Self._byte_swizzle_base, 4)
    ) if Self.enable_swizzle else Optional[Swizzle]()
    """Optional producer-side (byte-space) SMEM-store swizzle."""

    # Scheduling-relevant geometry (vmcnt-per-prefetch and lgkm-per-
    # frag-load) derived from the kernel's tile shape, dtype, and
    # 4-warp 2x2 quadrant layout. Consumed by the framework arm via
    # `PipelineConfig.vm_per_load_*` / `ScheduleConfig.lgkm_per_load_*`.
    comptime _geometry = Self._build_geometry()
    comptime VMCNT_PER_LOAD_A = Self._geometry.vm_per_load_a
    """Global-load vmcnt cost per A prefetch (from `KernelGeometry`)."""
    comptime VMCNT_PER_LOAD_B = Self._geometry.vm_per_load_b
    """Global-load vmcnt cost per B prefetch (from `KernelGeometry`)."""

    @staticmethod
    def _build_geometry() -> KernelGeometry:
        """Builds the `KernelGeometry` for this kernel's 4-warp 2x2 layout.

        Hard-codes the 4-wave kernel's structural assumptions:
          - 4 loading warps (= total_warps).
          - 2x2 warp grid in (M, N), so WM = BM/2 and WN = BN/2.
          - QuadrantMmaOp pattern: `(num_m_mmas, num_n_mmas)` split
            into 2x2 quadrants.
          - AMD `buffer_load_lds` issues 16-byte loads regardless of
            dtype, so `simd_width` is `16 / elem_bytes` rather than
            `simd_width_of[in_type]()` (the latter is host-target-
            dependent and gives wrong values when this factory is
            comptime-called from a CPU host).
        """
        comptime in_type = Self.in_type
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime MMA_K = Self.MMA_K
        comptime elem_bytes = size_of[in_type]()
        comptime simd_width = 16 // elem_bytes
        comptime is_fp8 = in_type.is_float8()
        comptime half_BM = BM // 2
        comptime half_BN = BN // 2
        comptime WM = BM // 2
        comptime WN = BN // 2
        comptime num_m_mmas = WM // MMA_M
        comptime num_n_mmas = WN // MMA_N
        comptime num_k_mmas = BK // MMA_K
        comptime quadrant_m_mmas = num_m_mmas // 2
        comptime quadrant_n_mmas = num_n_mmas // 2

        # === Tier 2 sanity asserts: every count above must be >= 1 ===
        # Without these, an unusual (BM, BN, BK, dtype) silently floors
        # one of these to 0 → kernel compiles, runs, produces garbage.
        comptime assert num_k_mmas >= 1, (
            "num_k_mmas = BK // MMA_K must be >= 1 (BK >= MMA_K). Got"
            " BK=, MMA_K="
        )
        comptime assert (
            num_m_mmas >= 1
        ), "num_m_mmas = WM // MMA_M must be >= 1 (WM = BM/2 >= MMA_M)."
        comptime assert (
            num_n_mmas >= 1
        ), "num_n_mmas = WN // MMA_N must be >= 1 (WN = BN/2 >= MMA_N)."
        comptime assert num_m_mmas >= 2 and num_m_mmas % 2 == 0, (
            "QuadrantMmaOp requires num_m_mmas >= 2 AND even"
            " (quadrant_m_mmas = num_m_mmas // 2 must be >= 1)."
        )
        comptime assert num_n_mmas >= 2 and num_n_mmas % 2 == 0, (
            "QuadrantMmaOp requires num_n_mmas >= 2 AND even"
            " (quadrant_n_mmas = num_n_mmas // 2 must be >= 1)."
        )

        # vm_per_load: number of distinct buffer_load_lds transactions
        # per prefetch tile, given 4 loading warps.
        #
        # `ceildiv` (not floor-div) so that under-supplied sub-tiles —
        # `half_BM < rows_per_iter_4warp` (e.g. BM=64+BK=32+bf16, where
        # 256 loading threads can transfer the whole 32x32 sub-tile in
        # ~half a wave) — get `vm_per_load_* == 1`, matching the loader's
        # ceildiv-based `num_iterations`. With floor-div both halves
        # rounded to 0 in lock-step: schedule emitted zero vmcnt drains
        # *and* loader emitted zero buffer_load_lds → silent garbage.
        comptime loads_per_row = BK // simd_width
        comptime assert loads_per_row >= 1 and BK % simd_width == 0, (
            "BK must be >= simd_width AND a multiple of simd_width"
            " (simd_width = 16 / size_of[in_type])."
        )
        comptime rows_per_iter_4warp = (4 * WARP_SIZE) // loads_per_row
        comptime vm_per_load_a = ceildiv(half_BM, rows_per_iter_4warp)
        comptime vm_per_load_b = ceildiv(half_BN, rows_per_iter_4warp)
        comptime assert vm_per_load_a >= 1, (
            "vm_per_load_a = ceildiv(half_BM, rows_per_iter_4warp) == 0"
            " — half_BM must be >= 1. Increase BM."
        )
        comptime assert vm_per_load_b >= 1, (
            "vm_per_load_b = ceildiv(half_BN, rows_per_iter_4warp) == 0"
            " — half_BN must be >= 1. Increase BN."
        )

        # lgkm_per_load: ds_reads per frag-load, accounting for FP8's
        # split-LDS path (16-byte fragments) vs BF16's full-fragment
        # path.
        comptime mma_frag_w = (MMA_M * MMA_K) // WARP_SIZE
        comptime assert mma_frag_w >= 1, (
            "mma_frag_w = (MMA_M * MMA_K) // WARP_SIZE must be >= 1."
            " MMA shape too small for the wavefront width."
        )
        comptime use_split_lds = is_fp8 and MMA_M == 16 and MMA_K == 128
        comptime lds_frag_w = 16 if use_split_lds else mma_frag_w
        comptime k_loads_per_mma = mma_frag_w // lds_frag_w
        comptime assert (
            k_loads_per_mma >= 1
        ), "k_loads_per_mma = mma_frag_w // lds_frag_w must be >= 1."
        comptime ds_reads_per_frag = ceildiv(lds_frag_w * elem_bytes, 16)
        comptime lgkm_per_load_a = (
            quadrant_m_mmas * num_k_mmas * k_loads_per_mma * ds_reads_per_frag
        )
        comptime lgkm_per_load_b = (
            quadrant_n_mmas * num_k_mmas * k_loads_per_mma * ds_reads_per_frag
        )
        comptime assert lgkm_per_load_a >= 1 and lgkm_per_load_b >= 1, (
            "lgkm_per_load_a/b must be >= 1 — frag-load wait derivation"
            " would otherwise compute zero lgkmcnt drains."
        )

        return KernelGeometry(
            BM=BM,
            BN=BN,
            BK=BK,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            elem_bytes=elem_bytes,
            simd_width=simd_width,
            is_fp8=is_fp8,
            vm_per_load_a=vm_per_load_a,
            vm_per_load_b=vm_per_load_b,
            lgkm_per_load_a=lgkm_per_load_a,
            lgkm_per_load_b=lgkm_per_load_b,
        )

    # Total SMEM the kernel allocates: 2 stages x 2 subtiles for each of
    # A and B, where A subtiles are (BM/2)*BK and B subtiles are
    # (BN/2)*BK in `in_type`. Collapses to 2*(BM+BN)*BK*size_of[in_type].
    comptime SMEM_BYTES = (
        2 * (Self.BM + Self.BN) * Self.BK * size_of[Self.in_type]()
    )
    """SMEM footprint per workgroup (bytes), derived from BM/BN/BK/in_type."""

    # Cap at the MI355X / gfx950 per-CU LDS budget (160 KB on CDNA4);
    # tile-shape picks should stay well under to leave headroom for
    # register allocation. Static `stack_allocation` works up to the
    # full 160 KB on CDNA4 — `amd_ping_pong_matmul` already runs at
    # 128 KB without `MAX_DYNAMIC_SHARED_SIZE_BYTES`.
    comptime _SMEM_LIMIT_BYTES = MI355X.shared_memory_per_multiprocessor

    @staticmethod
    def is_valid_config() -> Bool:
        """Returns whether the kernel's tile shapes are a viable config.

        Pure predicate, does not raise. Use this from autotune drivers
        and dispatcher fallbacks to filter out impossible (BM, BN, BK,
        dtype) combinations before instantiating the kernel. The full
        per-check set is in `validate_config`; this is its non-throwing
        counterpart.

        Returns:
            True if every structural and resource invariant holds.
        """
        return (
            Self.BM % Self.WM == 0
            and Self.BN % Self.WN == 0
            and Self.BK % Self.MMA_K == 0
            and Self.WM % Self.MMA_M == 0
            and Self.WN % Self.MMA_N == 0
            and Self.total_warps == 4
            and Self.num_warps_m == 2
            and Self.num_warps_n == 2
            and Self.num_m_mmas % 2 == 0
            and Self.num_n_mmas % 2 == 0
            and Self.a_type == Self.b_type
            and 16 % size_of[Self.in_type]() == 0
            and Self.SMEM_BYTES <= Self._SMEM_LIMIT_BYTES
        )

    @staticmethod
    def validate_config():
        """Asserts that the kernel's tile shapes meet 4-wave invariants.

        Throws (via `comptime assert`) on the first failing invariant,
        with a check-specific message. Called from `run` so any kernel
        instantiation that compiles is guaranteed valid. For non-
        throwing tests (autotune sweeps, dispatcher pre-flight), use
        `is_valid_config` instead.
        """
        comptime assert (
            Self.BM % Self.WM == 0
        ), "Block M must be divisible by Warp M"
        comptime assert (
            Self.BN % Self.WN == 0
        ), "Block N must be divisible by Warp N"
        comptime assert (
            Self.BK % Self.MMA_K == 0
        ), "Block K must be divisible by MMA K"
        comptime assert (
            Self.WM % Self.MMA_M == 0
        ), "Warp M must be divisible by MMA M"
        comptime assert (
            Self.WN % Self.MMA_N == 0
        ), "Warp N must be divisible by MMA N"
        comptime assert (
            Self.total_warps == 4
        ), "4-wave-simple kernel requires exactly 4 warps"
        comptime assert (
            Self.num_warps_m == 2
        ), "4-wave-simple requires 2 warps in M dimension"
        comptime assert (
            Self.num_warps_n == 2
        ), "4-wave-simple requires 2 warps in N dimension"
        comptime assert (
            Self.num_m_mmas % 2 == 0
        ), "4-wave-simple needs num_m_mmas even (QuadrantMmaOp constraint)"
        comptime assert (
            Self.num_n_mmas % 2 == 0
        ), "4-wave-simple needs num_n_mmas even (QuadrantMmaOp constraint)"
        comptime assert Self.a_type == Self.b_type, "A/B must match"
        comptime assert 16 % size_of[Self.in_type]() == 0, (
            "in_type element size must divide 16 (AMD buffer_load_lds"
            " transaction)"
        )
        comptime assert Self.SMEM_BYTES <= Self._SMEM_LIMIT_BYTES, (
            "SMEM budget exceeded: 2*(BM+BN)*BK*size_of[in_type] must fit in"
            " 64 KB"
        )

    @__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(1))
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @__name(
        t"amd_4wave_matmul_{Self.a_type}_{Self.b_type}_{Self.c_type}_BM{Self.BM}_BN{Self.BN}_BK{Self.BK}_WM{Self.WM}_WN{Self.WN}_SK{num_splits}"
    )
    @staticmethod
    def run[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
        c_engine: TensorEngine,
        *,
        num_splits: Int = 1,
    ](
        a: TileTensor[Self.a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    ):
        """Runs the 4-wave GEMM kernel for one workgroup tile.

        Emits the framework-driven body via `Pipeline4Wave` under
        `SchedulingStrategy.IDENTITY` + `minimal_barriers` +
        `omit_mma_set_prio`. The framework consumes the 24-op
        cross-stage-rotation body verbatim (no CSP/double-buffer
        reorder).

        Parameters:
            a_layout: Logical layout of `a`.
            b_layout: Logical layout of `b`.
            c_layout: Logical layout of `c`.
            a_engine: Engine of `a`.
            b_engine: Engine of `b`.
            c_engine: Engine of `c`.
            num_splits: Split-K factor (1 means no split).

        Args:
            a: Input tile-tensor for A.
            b: Input tile-tensor for B.
            c: Output tile-tensor for C (or workspace for split-K).
        """
        Self.validate_config()

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime simd_width = Self.simd_width
        comptime consumer_swizzle = Self.mma_swizzle
        comptime half_BM = Self.half_BM
        comptime half_BN = Self.half_BN
        comptime mma_tile_m = Self.mma_tile_m
        comptime mma_tile_n = Self.mma_tile_n

        var M = Int(a.dim[0]())
        comptime N = type_of(b).static_shape[0]
        comptime K = type_of(a).static_shape[1]
        comptime K_per_split = K // num_splits
        comptime assert (
            K_per_split * num_splits == K
        ), "num_splits must evenly divide K"
        comptime assert (
            K_per_split % (2 * Self.BK) == 0
        ), "K_per_split must be a multiple of 2*BK"

        # split_id from grid_dim.z; always 0 when num_splits=1
        # (grid_dim.z=1) so reading is harmless.
        var split_id = Int(block_idx.z)

        var _lane_id = lane_id()
        var _warp_id = readfirstlane(warp_id())
        # HK chiplet+L2 swizzle: 1D launch, derive (pid_m, pid_n) from
        # block_idx.x via XCD-stride remap + WGM=4 column-major group.
        # `num_pid_n` must match the launcher's `ceildiv(N, BN)` so that
        # the partial last column block (when N % BN != 0) decodes to its
        # own (pid_m, pid_n) instead of aliasing with another block. With
        # `N // BN` the partial block ID gets mis-mapped to the next
        # row's blocks, and the in-bounds tail of the partial block (cols
        # `(N//BN)*BN .. N-1`) silently goes unwritten.
        var num_pid_m = ceildiv(M, BM)
        comptime num_pid_n_static = ceildiv(N, BN)
        var pid_m_pid_n = _xcd_wgm_swizzle(
            Int(block_idx.x), num_pid_m, num_pid_n_static
        )
        var pid_m = pid_m_pid_n[0]
        var pid_n = pid_m_pid_n[1]
        var m = pid_m * BM
        var n = pid_n * BN
        var warp_id_m, warp_id_n = divmod(_warp_id, Self.num_warps_n)

        # === Unified-dtype GMEM views ===
        var a_gmem = a.bitcast[Self.a_type]()
        var b_gmem = b.bitcast[Self.in_type]()

        # === SMEM: 2 stages x 2 M-subtiles for A, 2 stages x 2 N-subtiles for B
        comptime a_half_layout = row_major[half_BM, BK]()
        var a_s0_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s0_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )

        comptime b_half_layout = row_major[half_BN, BK]()
        var b_s0_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s0_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )

        # MMA sub-tiles: each warp reads its mma_tile-sized slice from
        # BOTH M-halves (g0 and g1) for A, and BOTH N-halves (h0 and h1)
        # for B. The schedule's `subtile` axis indexes which M-half (for
        # A) or N-half (for B). Within a half, the warp picks its slice
        # via warp_id_m (or warp_id_n).
        var a_mma_s0_0 = a_s0_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s0_1 = a_s0_g1.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_0 = a_s1_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_1 = a_s1_g1.tile[mma_tile_m, BK](warp_id_m, 0)

        var b_mma_s0_0 = b_s0_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s0_1 = b_s0_h1.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_0 = b_s1_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_1 = b_s1_h1.tile[mma_tile_n, BK](warp_id_n, 0)

        # === DRAM->LDS loaders ===
        comptime _is_fp8 = Self.a_type.is_float8()
        comptime use_fp8_row_major = _is_fp8
        comptime byte_swizzle = Self.byte_swizzle

        # K-band selected by split_id; for num_splits=1 this is the
        # whole K range (split_id=0, K_per_split=K). The loaders now
        # source from the full A/B tensors and absorb the per-block
        # origin via `m_anchor`/`k_anchor`, so the SRDs' `num_records`
        # bound the actual allocation and the `load_tile` callsites use
        # within-block offsets — same convention as the conv
        # `TileLoaderLDSIm2col` sibling.
        var a_loader = TileLoaderLDS[
            Self.in_type,
            half_BM,
            BK,
            stride=type_of(a_gmem).static_stride[0],
            num_loading_warps=4,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](
            a_gmem,
            _warp_id,
            Int(_lane_id),
            m_anchor=pid_m * BM,
            k_anchor=split_id * K_per_split,
        )
        var b_loader = TileLoaderLDS[
            Self.in_type,
            half_BN,
            BK,
            stride=type_of(b_gmem).static_stride[0],
            num_loading_warps=4,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](
            b_gmem,
            _warp_id,
            Int(_lane_id),
            m_anchor=pid_n * BN,
            k_anchor=split_id * K_per_split,
        )

        # === MMA operator (full warp tile, quadrant access via methods) ===
        var mma_op = QuadrantMmaOp[
            out_type=Self.accum_dtype,
            in_type=Self.in_type,
            shape=Self.config.mma_shape,
            k_group_size=1,
            num_k_groups=Self.num_k_mmas,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            swizzle=consumer_swizzle,
        ]()

        @always_inline
        def s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        def s_setprio[priority: Int16]():
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        # === Load helpers ===
        var a_load_tiles = (
            (a_s0_g0, a_s0_g1),
            (a_s1_g0, a_s1_g1),
        )
        var b_load_tiles = (
            (b_s0_h0, b_s0_h1),
            (b_s1_h0, b_s1_h1),
        )

        @always_inline
        @__parameter
        def load_a[stage: Int, which: Int](k: Int):
            a_loader.load_tile(
                rebind[type_of(a_load_tiles[0][0])](
                    rebind[type_of(a_load_tiles[0])](a_load_tiles[stage])[which]
                ),
                m_offset=which * half_BM,
                k_offset=k,
            )

        @always_inline
        @__parameter
        def load_b[stage: Int, which: Int](k: Int):
            b_loader.load_tile(
                rebind[type_of(b_load_tiles[0][0])](
                    rebind[type_of(b_load_tiles[0])](b_load_tiles[stage])[which]
                ),
                m_offset=which * half_BN,
                k_offset=k,
            )

        # === MMA tile lookup: [stage][subtile] -> SMEM tile ===
        var a_mma_tiles = (
            (a_mma_s0_0, a_mma_s0_1),
            (a_mma_s1_0, a_mma_s1_1),
        )
        var b_mma_tiles = (
            (b_mma_s0_0, b_mma_s0_1),
            (b_mma_s1_0, b_mma_s1_1),
        )

        # ====================================================
        # Body: framework-driven via Pipeline4Wave.
        # ====================================================
        @__parameter
        @always_inline
        def _emit_framework_body():
            # `Pipeline4Wave.__init__` forces IDENTITY +
            # minimal_barriers + omit_mma_set_prio; the ScheduleConfig
            # we pass here only contributes `sched_barrier_mask` and
            # auto-waits opt-in (the rest is overridden by the schedule
            # struct).
            #
            # `sched_barrier_mask` is `0xFF` at BM>=128 (8 fences per
            # loop iter at MMA-block boundaries, matching
            # `_sched_barrier()`'s density) and `0` at BM<128 — fences
            # regress ~4% on BM=64 because the mini-iter is too small
            # to benefit.
            #
            # `wrap_waits_with_sched_barrier` is enabled at BM>=128 so
            # every wait/barrier group inside a block gets wrapped with
            # `schedule_barrier` fences — matching the density of
            # `_sched_barrier()` calls in the hand-tuned body. At
            # BM=64 the mini-iter is too small for the fences to help
            # (LLVM's default scheduler already does well there).
            # FP8 takes the split-LDS path (`use_split_lds=True` in
            # `_build_geometry`): each MMA fragment is split across two
            # ds_reads of 16-byte chunks. The dense fence hints below
            # (calibrated for bf16/fp16's single-frag mini-iter) confuse
            # ROCm 7.2.3's machine scheduler on that split pattern,
            # producing wrong output at tall-skinny shapes (e.g. M=4096
            # N=128 K=256 fp8 — 1006/524288 cells wrong). bf16 and fp16
            # at the same shape pass cleanly, confirming the issue is
            # fp8 + split-LDS-specific. Drop the fence hints for fp8 so
            # the scheduler treats it like the BM<128 path (no hints,
            # default LLVM scheduling).
            comptime _scheduled_use_fences = (
                Self.BM >= 128 and not Self.in_type.is_float8()
            )
            comptime SCHED_MASK = 0xFF if _scheduled_use_fences else 0
            comptime sched_config = ScheduleConfig(
                scheduling=SchedulingStrategy.IDENTITY,
                auto_waits=True,
                sched_barrier_mask=SCHED_MASK,
                wrap_waits_with_sched_barrier=_scheduled_use_fences,
            )
            comptime target = mi355x_target(
                vm_per_load_a=Self.VMCNT_PER_LOAD_A,
                vm_per_load_b=Self.VMCNT_PER_LOAD_B,
                max_globals=0 if (
                    Self.num_k_mmas
                    * Self.quadrant_m_mmas
                    * Self.quadrant_n_mmas
                    < 8
                ) else derive_safe_max_globals(Self.num_k_mmas),
            )
            # `KernelGeometry` bundles `is_fp8` + `lgkm_per_load_*` into
            # one comptime arg so they don't have to be threaded as
            # individual template params.
            comptime schedule = build_schedule[geometry=Self._geometry](
                sched_config, target
            )

            @__parameter
            @always_inline
            def _bind[entry: ScheduleEntry](k_base: Int):
                # Framework-level infrastructure tags (BARRIER / WAIT_* /
                # SET_PRIO / SCHEDULE_BARRIER) emit AMD intrinsics
                # directly; kernel-specific data tags (LOAD_A / LOAD_B /
                # MMA_LOAD_* / MMA) call the kernel's own loaders.
                comptime if entry.op.tag == _Ops.BARRIER.value:
                    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
                elif entry.op.tag == _Ops.WAIT_VM.value:
                    s_waitcnt[vmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.WAIT_LGKM.value:
                    s_waitcnt[lgkmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.SET_PRIO.value:
                    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](
                        Int16(entry.op.wait_value)
                    )
                elif entry.op.tag == _Ops.SCHEDULE_BARRIER.value:
                    schedule_barrier()
                else:
                    comptime k_off = entry.op.k_offset.signed_bk_multiple()
                    var k = k_base + k_off * BK
                    comptime if entry.op.tag == LOAD_A:
                        load_a[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == LOAD_B:
                        load_b[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == MMA_LOAD_A:
                        mma_op.load_a_quadrant[entry.op.subtile](
                            rebind[type_of(a_mma_tiles[0][0])](
                                rebind[type_of(a_mma_tiles[0])](
                                    a_mma_tiles[entry.op.stage]
                                )[entry.op.subtile]
                            )
                        )
                    elif entry.op.tag == MMA_LOAD_B:
                        mma_op.load_b_quadrant[entry.op.subtile](
                            rebind[type_of(b_mma_tiles[0][0])](
                                rebind[type_of(b_mma_tiles[0])](
                                    b_mma_tiles[entry.op.stage]
                                )[entry.op.subtile]
                            )
                        )
                    elif entry.op.tag == MMA:
                        mma_op.mma_quadrant[entry.op.stage, entry.op.subtile]()

            # Prologue. The schedule overrides `warp_stagger=none()`
            # (4-wave has only 4 warps, so warp-group staggering is
            # meaningless), sets `partial_prologue_drain=True`, and
            # supplies `bootstrap_frags()` for the two same-stage
            # sub=0 frag-loads needed for the first main iter. The
            # framework appends them to the prologue paired with
            # `wait_vm(N) + barrier` partial drains computed from
            # cumulative prefetch vm_cost. A single comptime for over
            # the full prologue emits everything correctly.
            comptime for i in range(len(schedule.prologue)):
                _bind[schedule.prologue[i]](0)

            # Main loop: step 2*BK because each schedule iter unrolls 2
            # source K-iters (matches ping-pong's stepping).
            for k in range(BK * 2, K_per_split, BK * 2):
                comptime for i in range(len(schedule.kernel)):
                    _bind[schedule.kernel[i]](k)

            # The schedule leaves several buffer_load_lds prefetches in flight
            # into the LDS double-buffer at the main-loop-to-epilogue boundary,
            # counting on MMA-cycle latency to hide them before the epilogue's
            # ds_read. Two cases erase that cushion: split-K launches
            # grid_dim.z = num_splits extra workgroups so occupancy is high and
            # per-workgroup latency hiding collapses, and K_per_split == 2*BK
            # leaves no main loop so the prologue's partial drain runs straight
            # into the epilogue with prefetches still in flight. Either way the
            # epilogue can read LDS slots before the writes land, corrupting a
            # few ULPs as run-to-run non-determinism. Drain VMEM and LDS once
            # here; the num_splits == 1 loop case keeps the cushion and compiles
            # this away.
            comptime _num_K_iters_static = K_per_split // (2 * BK)
            comptime if _num_K_iters_static == 1 or num_splits > 1:
                s_waitcnt[vmcnt=UInt32(0)]()
                s_waitcnt[lgkmcnt=UInt32(0)]()
                s_barrier()

            # Epilogue.
            comptime for i in range(len(schedule.epilogue)):
                _bind[schedule.epilogue[i]](K_per_split)

        _emit_framework_body()

        # === Output Store (shared) ===
        # 4-wave's warp output is interleaved: each warp covers 2
        # disjoint M-row ranges (one per M-half) × 2 disjoint N-col
        # ranges. Per (m_mma, n_mma), each lane writes a SIMD-of-
        # c_frag_size at a per-lane (m_global, n_global) coordinate
        # in the (BM × BN) block:
        #
        #   m_global = pid_m*BM + m_quad*half_BM
        #            + warp_id_m*mma_tile_m + m_within*MMA_M + thread_m
        #   n_global = pid_n*BN + n_quad*half_BN
        #            + warp_id_n*mma_tile_n + n_within*MMA_N
        #            + lane_group*c_frag_size
        #
        # `RegTileEpilogue` handles the per-lane chunk store and the
        # partial-chunk fallback at the column boundary; the kernel
        # only computes (m, n) and feeds the SIMD chunk. For split-K
        # the matmul kernel writes f32 partials to a stacked workspace
        # at row `split_id * M + pid_m * BM + m_global_base` while the
        # in-bounds gate uses the LOGICAL row `m + m_global_base` — the
        # two coincide for non-split-K. Lambda mode requires the lambda
        # to fire on the FINAL value, so split-K + lambda is rejected
        # at comptime; the reduce kernel applies the lambda there.
        comptime quad_m_mmas = Self.quadrant_m_mmas
        comptime quad_n_mmas = Self.quadrant_n_mmas

        comptime if Bool(Self.elementwise_lambda_fn):
            comptime assert num_splits == 1, (
                "elementwise epilogue is not supported with split-K"
                " (would fire on each partial, not the reduced output)"
            )

        var c_writer = RegTileEpilogue[
            Self.c_type,
            c_frag_size,
            elementwise_lambda_fn=Self.elementwise_lambda_fn,
        ](c)

        if m < M and n < N:
            var c_reg = mma_op.accum_tile()
            var lane_group, thread_m = divmod(Int(_lane_id), MMA_M)

            comptime for m_mma in range(num_m_mmas):
                comptime m_quad, m_within = divmod(m_mma, quad_m_mmas)
                var m_global_base = (
                    m_quad * half_BM
                    + Int(warp_id_m) * mma_tile_m
                    + m_within * MMA_M
                    + thread_m
                )
                var m_dram = split_id * M + pid_m * BM + m_global_base
                var m_logical = m + m_global_base
                if m_logical < M:
                    comptime for n_mma in range(num_n_mmas):
                        comptime n_quad, n_within = divmod(n_mma, quad_n_mmas)
                        var n_global = (
                            n
                            + n_quad * half_BN
                            + Int(warp_id_n) * mma_tile_n
                            + n_within * MMA_N
                            + lane_group * c_frag_size
                        )
                        var v = (
                            c_reg.tile[1, c_frag_size](m_mma, n_mma)
                            .raw_load[width=c_frag_size](0)
                            .cast[Self.c_type]()
                        )
                        c_writer.store(v, m=m_dram, n=n_global)

    @__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(1))
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @__name(
        t"amd_4wave_conv2d_{Self.a_type}_{Self.b_type}_{Self.c_type}_BM{Self.BM}_BN{Self.BN}_BK{Self.BK}_WM{Self.WM}_WN{Self.WN}_res{has_residual}"
    )
    @staticmethod
    def run_conv2d[
        conv_config: Conv2DKernelConfig,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        has_residual: Bool = False,
        # Pre-residual fused compute lambda (SM100 reference:
        # `D = lambda(Conv(A,B)) + beta * C`). Fires on the post-cast
        # `c_type` MMA output, BEFORE the residual FMA. Signature
        # `(IndexList[2], SIMD) capturing -> SIMD`. Use for bias / ReLU /
        # SiLU / GELU fusion. The post-residual store-site lambda is
        # the existing struct-level `Self.elementwise_lambda_fn` and
        # fires after the residual FMA (or directly on the MMA output
        # when `has_residual=False`).
        elementwise_compute_lambda_fn: Optional[
            elementwise_compute_lambda_type
        ] = None,
    ](
        a: TileTensor[Self.a_type, a_layout, ImmutAnyOrigin],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin],
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin],
        source_ptr: UnsafePointer[Scalar[Self.c_type], ImmutAnyOrigin],
        source_row_stride: Int32,
        beta: Float32,
    ):
        """Runs the 4-wave kernel as a 2D convolution via implicit-GEMM.

        Sibling of `run()` (the matmul entry point); both share the
        4-warp 2x2 quadrant layout, the same MFMA shapes, and the same
        software-pipeline schedule; only the A-operand loader differs.
        `run()` uses `TileLoaderLDS` (linear MK source); this method
        uses `TileLoaderLDSIm2col`, which materializes the A operand
        from a 4D NHWC input via in-line im2col addressing.

        Optional in-kernel residual prefetch via `has_residual`. When
        True, the kernel bulk-prefetches `source` (a 2D
        `[M, C_out]`-aliased view of an NHWC residual buffer) into VGPRs
        at the start of the epilogue. By the time the FMA-and-store
        loop runs, all 32 per-lane residual loads are in flight in
        parallel, replacing the per-store
        `global_load → wait → store` staircase that costs ~24% on
        memory-bound shapes. The launcher passes the residual pointer,
        row stride, and `beta` scale; the kernel applies
        `out = mma + beta * residual` in Float32 and casts back to
        `c_type` for the store.

        Parameters:
            conv_config: Conv geometry (filter shape, stride, dilation,
                pad, input H/W, runtime-HW flag).
            a_layout: Logical layout of `a` (4D NHWC).
            b_layout: Logical layout of `b` (2D `[C_out, K]` filter).
            c_layout: Logical layout of `c` (2D `[M, C_out]` output).
            has_residual: When True, prefetch + FMA in `source * beta`
                during the epilogue. When False, the residual args are
                unused and the epilogue is identical to the no-residual
                kernel (dead-code-eliminated by the compiler).
            elementwise_compute_lambda_fn: Optional pre-residual fused
                compute lambda. Fires on the post-cast `c_type` MMA
                output BEFORE the residual FMA, matching the SM100
                `D = lambda(Conv(A,B)) + beta * C` ordering. Use for
                bias / ReLU / SiLU / GELU fusion. Signature
                `(IndexList[2], SIMD) capturing -> SIMD`. When unset,
                the comptime branch dead-code-eliminates.

        Args:
            a: Input tile-tensor for A (NHWC activations).
            b: Input tile-tensor for B (filter, RSCF / FCRS shape).
            c: Output tile-tensor for C (flattened M x C_out view).
            source_ptr: Pointer to the residual buffer (2D
                `[M, C_out]`-aliased view of NHWC). Unused when
                `has_residual=False`; the launcher passes a null sentinel
                in that case.
            source_row_stride: Element stride of the residual's row
                dimension. Unused when `has_residual=False`.
            beta: Residual scale. Unused when `has_residual=False`.
        """
        Self.validate_config()
        var _source_row_stride = Int(source_row_stride)

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime simd_width = Self.simd_width
        comptime consumer_swizzle = Self.mma_swizzle
        comptime half_BM = Self.half_BM
        comptime half_BN = Self.half_BN
        comptime mma_tile_m = Self.mma_tile_m
        comptime mma_tile_n = Self.mma_tile_n
        # Filter spatial volume: R*S in 2D, Q*R*S in 3D. Used as the
        # divisor that maps the filter buffer's K dim back to C_in.
        comptime _is_3d_conv = conv_config.Q > 1
        comptime conv_RS = conv_config.R * conv_config.S
        comptime conv_RS_filter = (
            conv_config.Q * conv_RS if _is_3d_conv else conv_RS
        )
        comptime num_splits = 1  # conv2d does not support split-K

        # Conv adaptation: A is 4D NHWC, C is the 2D NHW×Cout output view.
        # M (GEMM) = N*H*W; read from C's row count. N (GEMM) = Cout (B rows).
        # K (GEMM) = K_filter (possibly K-padded; real K is R*S*C_in_used).
        var M = Int(c.dim[0]())
        comptime N = type_of(b).static_shape[0]
        comptime K = type_of(b).static_shape[1]
        # K-padding support: when conv_config.C_in > 0, the caller has zero-padded
        # the filter's trailing K to bring K_filter up to a multiple of
        # 2*BK. The "real" C_in is conv_config.C_in; the kernel uses it to set
        # the loader's C dimension. The padded K rows in the filter
        # contribute 0 to the MMA so the output is correct, and the A
        # loader's reads in the padded K range either hit the SRD-OOB
        # clamp (returning 0) or read "wrong but valid" input data (also
        # multiplied by 0 in the MMA).
        comptime C_in_used = (
            conv_config.C_in if conv_config.C_in > 0 else K // conv_RS_filter
        )
        comptime K_per_split = K // num_splits
        comptime assert (
            K_per_split * num_splits == K
        ), "num_splits must evenly divide K"
        comptime assert (
            K_per_split % (2 * Self.BK) == 0
        ), "K_per_split must be a multiple of 2*BK"
        comptime assert num_splits == 1, "amd_4wave_conv requires num_splits=1"
        comptime assert conv_RS_filter * C_in_used <= K, (
            "Filter K dim must be >= Q*R*S*C_in (caller K-pads to a multiple"
            " of 2*BK = 256 when Q*R*S*C_in is not aligned; trailing K must"
            " be zero-filled by the caller)"
        )

        # split_id from grid_dim.z; always 0 when num_splits=1
        # (grid_dim.z=1) so reading is harmless.
        var split_id = Int(block_idx.z)

        var _lane_id = lane_id()
        var _warp_id = readfirstlane(warp_id())
        # HK chiplet+L2 swizzle: 1D launch, derive (pid_m, pid_n) from
        # block_idx.x via XCD-stride remap + WGM=4 column-major group.
        # `num_pid_n` must match the launcher's `ceildiv(N, BN)` so that
        # the partial last column block (when N % BN != 0) decodes to its
        # own (pid_m, pid_n) instead of aliasing with another block. With
        # `N // BN` the partial block ID gets mis-mapped to the next
        # row's blocks, and the in-bounds tail of the partial block (cols
        # `(N//BN)*BN .. N-1`) silently goes unwritten.
        var num_pid_m = ceildiv(M, BM)
        comptime num_pid_n_static = ceildiv(N, BN)
        var pid_m_pid_n = _xcd_wgm_swizzle(
            Int(block_idx.x), num_pid_m, num_pid_n_static
        )
        var pid_m = pid_m_pid_n[0]
        var pid_n = pid_m_pid_n[1]
        var m = pid_m * BM
        var n = pid_n * BN
        var warp_id_m, warp_id_n = divmod(_warp_id, Self.num_warps_n)

        # === Unified-dtype GMEM views ===
        # A is 4D NHWC; the conv loader handles (m, k) -> NHWC translation
        # directly. No bitcast needed (input dtype matches Self.a_type).
        var b_gmem = b.bitcast[Self.in_type]()

        # === SMEM: 2 stages x 2 M-subtiles for A, 2 stages x 2 N-subtiles for B
        comptime a_half_layout = row_major[half_BM, BK]()
        var a_s0_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s0_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )

        comptime b_half_layout = row_major[half_BN, BK]()
        var b_s0_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s0_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )

        # MMA sub-tiles: each warp reads its mma_tile-sized slice from
        # BOTH M-halves (g0 and g1) for A, and BOTH N-halves (h0 and h1)
        # for B. The schedule's `subtile` axis indexes which M-half (for
        # A) or N-half (for B). Within a half, the warp picks its slice
        # via warp_id_m (or warp_id_n).
        var a_mma_s0_0 = a_s0_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s0_1 = a_s0_g1.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_0 = a_s1_g0.tile[mma_tile_m, BK](warp_id_m, 0)
        var a_mma_s1_1 = a_s1_g1.tile[mma_tile_m, BK](warp_id_m, 0)

        var b_mma_s0_0 = b_s0_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s0_1 = b_s0_h1.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_0 = b_s1_h0.tile[mma_tile_n, BK](warp_id_n, 0)
        var b_mma_s1_1 = b_s1_h1.tile[mma_tile_n, BK](warp_id_n, 0)

        # === DRAM->LDS loaders ===
        comptime _is_fp8 = Self.a_type.is_float8()
        comptime use_fp8_row_major = _is_fp8
        comptime byte_swizzle = Self.byte_swizzle

        # A-loader: conv-aware variant of TileLoaderLDS. Takes the full
        # 4D NHWC tensor (SRD covers the whole input). The per-block
        # origin is absorbed into the loader via `m_anchor`, so the
        # `load_a` closure passes within-block offsets — same calling
        # convention as the matmul `TileLoaderLDS` sibling. For
        # milestone 1 (R=S=1) the address math collapses to `m * C + k`,
        # identical to TileLoaderLDS with stride=C.
        comptime _LoaderTy = TileLoaderLDSIm2col[
            Self.in_type,
            half_BM,
            BK,
            C_in_used,
            num_loading_warps=4,
            H=conv_config.H,
            W=conv_config.W,
            H_out=conv_config.H_out,
            W_out=conv_config.W_out,
            R=conv_config.R,
            S=conv_config.S,
            stride_h=conv_config.stride_h,
            stride_w=conv_config.stride_w,
            dilation_h=conv_config.dilation_h,
            dilation_w=conv_config.dilation_w,
            pad_h=conv_config.pad_h,
            pad_w=conv_config.pad_w,
            # 3D mode (Q>1) activates the NDHWC im2col path inside the
            # loader. 2D callers leave these at the Q=1 defaults so the
            # loader compiles identically to before.
            Q=conv_config.Q,
            D=conv_config.D,
            D_out=conv_config.D_out,
            stride_d=conv_config.stride_d,
            dilation_d=conv_config.dilation_d,
            pad_d=conv_config.pad_d,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
            use_runtime_hw=conv_config.use_runtime_hw,
        ]
        var a_loader: _LoaderTy
        comptime if conv_config.use_runtime_hw:
            # Read runtime conv geometry from the input tile-tensor.
            # 2D: dim[1]=H, dim[2]=W. 3D: dim[1]=D, dim[2]=H, dim[3]=W.
            # h_out / w_out (and d_out in 3D) are derived from input
            # spatial dims + conv params.
            comptime _eff_R = conv_config.dilation_h * (conv_config.R - 1) + 1
            comptime _eff_S = conv_config.dilation_w * (conv_config.S - 1) + 1
            comptime if _is_3d_conv:
                var _rt_d = Int(a.dim[1]())
                var _rt_h = Int(a.dim[2]())
                var _rt_w = Int(a.dim[3]())
                comptime _eff_Q = (
                    conv_config.dilation_d * (conv_config.Q - 1) + 1
                )
                var _rt_d_out = (
                    _rt_d + 2 * conv_config.pad_d - _eff_Q
                ) // conv_config.stride_d + 1
                var _rt_h_out = (
                    _rt_h + 2 * conv_config.pad_h - _eff_R
                ) // conv_config.stride_h + 1
                var _rt_w_out = (
                    _rt_w + 2 * conv_config.pad_w - _eff_S
                ) // conv_config.stride_w + 1
                a_loader = _LoaderTy(
                    a,
                    _warp_id,
                    Int(_lane_id),
                    runtime_d=_rt_d,
                    runtime_h=_rt_h,
                    runtime_w=_rt_w,
                    runtime_d_out=_rt_d_out,
                    runtime_h_out=_rt_h_out,
                    runtime_w_out=_rt_w_out,
                    m_anchor=pid_m * BM,
                )
            else:
                var _rt_h = Int(a.dim[1]())
                var _rt_w = Int(a.dim[2]())
                var _rt_h_out = (
                    _rt_h + 2 * conv_config.pad_h - _eff_R
                ) // conv_config.stride_h + 1
                var _rt_w_out = (
                    _rt_w + 2 * conv_config.pad_w - _eff_S
                ) // conv_config.stride_w + 1
                a_loader = _LoaderTy(
                    a,
                    _warp_id,
                    Int(_lane_id),
                    runtime_h=_rt_h,
                    runtime_w=_rt_w,
                    runtime_h_out=_rt_h_out,
                    runtime_w_out=_rt_w_out,
                    m_anchor=pid_m * BM,
                )
        else:
            a_loader = _LoaderTy(
                a, _warp_id, Int(_lane_id), m_anchor=pid_m * BM
            )
        # B-loader: full-filter SRD with per-block anchor — same calling
        # convention as the matmul's B-loader (filter is the matmul's A
        # in NK shape; conv just relabels).
        var b_loader = TileLoaderLDS[
            Self.in_type,
            half_BN,
            BK,
            stride=type_of(b_gmem).static_stride[0],
            num_loading_warps=4,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](
            b_gmem,
            _warp_id,
            Int(_lane_id),
            m_anchor=pid_n * BN,
            k_anchor=split_id * K_per_split,
        )

        # === MMA operator (full warp tile, quadrant access via methods) ===
        var mma_op = QuadrantMmaOp[
            out_type=Self.accum_dtype,
            in_type=Self.in_type,
            shape=Self.config.mma_shape,
            k_group_size=1,
            num_k_groups=Self.num_k_mmas,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            swizzle=consumer_swizzle,
        ]()

        @always_inline
        def s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        def s_setprio[priority: Int16]():
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        # === Load helpers ===
        var a_load_tiles = (
            (a_s0_g0, a_s0_g1),
            (a_s1_g0, a_s1_g1),
        )
        var b_load_tiles = (
            (b_s0_h0, b_s0_h1),
            (b_s1_h0, b_s1_h1),
        )

        @always_inline
        @__parameter
        def load_a[stage: Int, which: Int](k: Int):
            # `m_anchor=pid_m*BM` baked into the loader at construction;
            # callsite only carries the within-block `which*half_BM`
            # offset — same shape as the matmul's load_a / load_b.
            a_loader.load_tile(
                rebind[type_of(a_load_tiles[0][0])](
                    rebind[type_of(a_load_tiles[0])](a_load_tiles[stage])[which]
                ),
                m_offset=which * half_BM,
                k_offset=k,
            )

        @always_inline
        @__parameter
        def load_b[stage: Int, which: Int](k: Int):
            b_loader.load_tile(
                rebind[type_of(b_load_tiles[0][0])](
                    rebind[type_of(b_load_tiles[0])](b_load_tiles[stage])[which]
                ),
                m_offset=which * half_BN,
                k_offset=k,
            )

        # === MMA tile lookup: [stage][subtile] -> SMEM tile ===
        var a_mma_tiles = (
            (a_mma_s0_0, a_mma_s0_1),
            (a_mma_s1_0, a_mma_s1_1),
        )
        var b_mma_tiles = (
            (b_mma_s0_0, b_mma_s0_1),
            (b_mma_s1_0, b_mma_s1_1),
        )

        # ====================================================
        # Body: framework-driven via Pipeline4Wave.
        # ====================================================
        @__parameter
        @always_inline
        def _emit_framework_body():
            # `Pipeline4Wave.__init__` forces IDENTITY +
            # minimal_barriers + omit_mma_set_prio; the ScheduleConfig
            # we pass here only contributes `sched_barrier_mask` and
            # auto-waits opt-in (the rest is overridden by the schedule
            # struct).
            #
            # `sched_barrier_mask` is `0xFF` at BM>=128 (8 fences per
            # loop iter at MMA-block boundaries, matching
            # `_sched_barrier()`'s density) and `0` at BM<128 — fences
            # regress ~4% on BM=64 because the mini-iter is too small
            # to benefit.
            #
            # `wrap_waits_with_sched_barrier` is enabled at BM>=128 so
            # every wait/barrier group inside a block gets wrapped with
            # `schedule_barrier` fences — matching the density of
            # `_sched_barrier()` calls in the hand-tuned body. At
            # BM=64 the mini-iter is too small for the fences to help
            # (LLVM's default scheduler already does well there).
            comptime SCHED_MASK = 0xFF if Self.BM >= 128 else 0
            comptime sched_config = ScheduleConfig(
                scheduling=SchedulingStrategy.IDENTITY,
                auto_waits=True,
                sched_barrier_mask=SCHED_MASK,
                wrap_waits_with_sched_barrier=Self.BM >= 128,
            )
            comptime target = mi355x_target(
                vm_per_load_a=Self.VMCNT_PER_LOAD_A,
                vm_per_load_b=Self.VMCNT_PER_LOAD_B,
                max_globals=0 if (
                    Self.num_k_mmas
                    * Self.quadrant_m_mmas
                    * Self.quadrant_n_mmas
                    < 8
                ) else derive_safe_max_globals(Self.num_k_mmas),
            )
            # `KernelGeometry` bundles `is_fp8` + `lgkm_per_load_*` into
            # one comptime arg so they don't have to be threaded as
            # individual template params.
            comptime schedule = build_schedule[geometry=Self._geometry](
                sched_config, target
            )

            @__parameter
            @always_inline
            def _bind[entry: ScheduleEntry](k_base: Int):
                # Framework-level infrastructure tags (BARRIER / WAIT_* /
                # SET_PRIO / SCHEDULE_BARRIER) emit AMD intrinsics
                # directly; kernel-specific data tags (LOAD_A / LOAD_B /
                # MMA_LOAD_* / MMA) call the kernel's own loaders.
                comptime if entry.op.tag == _Ops.BARRIER.value:
                    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
                elif entry.op.tag == _Ops.WAIT_VM.value:
                    s_waitcnt[vmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.WAIT_LGKM.value:
                    s_waitcnt[lgkmcnt=UInt32(entry.op.wait_value)]()
                elif entry.op.tag == _Ops.SET_PRIO.value:
                    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](
                        Int16(entry.op.wait_value)
                    )
                elif entry.op.tag == _Ops.SCHEDULE_BARRIER.value:
                    schedule_barrier()
                else:
                    comptime k_off = entry.op.k_offset.signed_bk_multiple()
                    var k = k_base + k_off * BK
                    comptime if entry.op.tag == LOAD_A:
                        load_a[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == LOAD_B:
                        load_b[entry.op.stage, entry.op.subtile](k)
                    elif entry.op.tag == MMA_LOAD_A:
                        mma_op.load_a_quadrant[entry.op.subtile](
                            rebind[type_of(a_mma_tiles[0][0])](
                                rebind[type_of(a_mma_tiles[0])](
                                    a_mma_tiles[entry.op.stage]
                                )[entry.op.subtile]
                            )
                        )
                    elif entry.op.tag == MMA_LOAD_B:
                        mma_op.load_b_quadrant[entry.op.subtile](
                            rebind[type_of(b_mma_tiles[0][0])](
                                rebind[type_of(b_mma_tiles[0])](
                                    b_mma_tiles[entry.op.stage]
                                )[entry.op.subtile]
                            )
                        )
                    elif entry.op.tag == MMA:
                        mma_op.mma_quadrant[entry.op.stage, entry.op.subtile]()

            # Prologue. The schedule overrides `warp_stagger=none()`
            # (4-wave has only 4 warps, so warp-group staggering is
            # meaningless), sets `partial_prologue_drain=True`, and
            # supplies `bootstrap_frags()` for the two same-stage
            # sub=0 frag-loads needed for the first main iter. The
            # framework appends them to the prologue paired with
            # `wait_vm(N) + barrier` partial drains computed from
            # cumulative prefetch vm_cost. A single comptime for over
            # the full prologue emits everything correctly.
            comptime for i in range(len(schedule.prologue)):
                _bind[schedule.prologue[i]](0)

            # See the matching block in `run()` for the full rationale.
            # When `K_per_split == 2*BK` the main loop runs zero times
            # and the framework epilogue starts immediately after the
            # prologue's `partial_prologue_drain`, with 6 prefetches
            # still in flight. The first epilogue block's `ds_read`
            # races with those `buffer_load_lds → LDS` writes (which
            # the per-block `wait_vm[0]` only drains AFTER the
            # frag-load issues). Mirror the handwritten body's
            # top-of-iter sync to drain everything before the
            # epilogue's first `ds_read` fires.
            comptime _num_K_iters_static = K_per_split // (2 * BK)
            comptime if _num_K_iters_static == 1:
                s_waitcnt[vmcnt=UInt32(0)]()
                s_waitcnt[lgkmcnt=UInt32(0)]()
                s_barrier()

            # Main loop: step 2*BK because each schedule iter unrolls 2
            # source K-iters (matches ping-pong's stepping).
            for k in range(BK * 2, K_per_split, BK * 2):
                comptime for i in range(len(schedule.kernel)):
                    _bind[schedule.kernel[i]](k)

            # Epilogue.
            comptime for i in range(len(schedule.epilogue)):
                _bind[schedule.epilogue[i]](K_per_split)

        # === Output Store (shared) ===
        # 4-wave's warp output is interleaved: each warp covers 2
        # disjoint M-row ranges (one per M-half) × 2 disjoint N-col
        # ranges. Per (m_mma, n_mma), each lane writes a SIMD-of-
        # c_frag_size at a per-lane (m_global, n_global) coordinate
        # in the (BM × BN) block:
        #
        #   m_global = pid_m*BM + m_quad*half_BM
        #            + warp_id_m*mma_tile_m + m_within*MMA_M + thread_m
        #   n_global = pid_n*BN + n_quad*half_BN
        #            + warp_id_n*mma_tile_n + n_within*MMA_N
        #            + lane_group*c_frag_size
        #
        # `RegTileEpilogue` handles the per-lane chunk store and the
        # partial-chunk fallback at the column boundary; the kernel
        # only computes (m, n) and feeds the SIMD chunk. For split-K
        # the matmul kernel writes f32 partials to a stacked workspace
        # at row `split_id * M + pid_m * BM + m_global_base` while the
        # in-bounds gate uses the LOGICAL row `m + m_global_base` — the
        # two coincide for non-split-K. Lambda mode requires the lambda
        # to fire on the FINAL value, so split-K + lambda is rejected
        # at comptime; the reduce kernel applies the lambda there.
        comptime quad_m_mmas = Self.quadrant_m_mmas
        comptime quad_n_mmas = Self.quadrant_n_mmas

        comptime if Bool(Self.elementwise_lambda_fn):
            comptime assert num_splits == 1, (
                "elementwise epilogue is not supported with split-K"
                " (would fire on each partial, not the reduced output)"
            )

        # Residual prefetch: when `has_residual`, bulk-issue per-lane
        # `buffer_load_*` for source[m_logical, n_global] into a VGPR
        # array BEFORE `_emit_framework_body()` runs. The HBM loads then
        # overlap with the entire main loop's MFMAs instead of being
        # exposed in the epilogue (~7-15 pp recovery for memory-bound
        # 1×1 / small-K shapes, where the epilogue is a larger fraction
        # of total kernel time). A single `s_waitcnt vmcnt(0)` drains
        # the cluster before the first `v_pk_fma_f32` in the epilogue.
        #
        # Storage: per-lane `Array` of
        # `SIMD[c_type, c_frag_size]` × num_m_mmas × num_n_mmas. For
        # BM=BN=128 / MMA=16x16 / c_frag_size=4 that's 16 slots × 4 bf16
        # = 16 dwords per lane (well under the 196-Dword VGPR headroom
        # at 316 baseline). OOB blocks waste a few HBM reads — SRD
        # bounds clamping returns 0, so it's harmless.
        comptime n_slots = num_m_mmas * num_n_mmas
        var prefetched = Array[SIMD[Self.c_type, c_frag_size], n_slots](
            uninitialized=True
        )
        var lane_group, thread_m = divmod(Int(_lane_id), MMA_M)

        comptime if has_residual:
            # SRD covers the full source buffer in *elements* (the
            # constructor multiplies by `size_of[dtype]()` internally).
            # The `load` API likewise takes its `vector_offset` argument
            # in elements and multiplies by `size_of[dtype]()` to derive
            # the buffer byte offset — see `AMDBufferResource.load` in
            # `max.gpu.intrinsics`. Both sides must agree: passing a
            # byte-scaled `num_records` paired with a byte-scaled
            # `vector_offset` doubles the effective stride (the bug
            # this fix replaces), making workgroups with
            # `pid_m >= num_pid_m/2` read OOB → SRD-clamped to 0.
            # Keep both expressed in elements.
            var src_size_elem = M * _source_row_stride
            var src_bc = AMDBufferResource(
                readfirstlane(source_ptr), readfirstlane(src_size_elem)
            )

            comptime for m_mma in range(num_m_mmas):
                comptime m_quad, m_within = divmod(m_mma, quad_m_mmas)
                var m_global_base = (
                    m_quad * half_BM
                    + Int(warp_id_m) * mma_tile_m
                    + m_within * MMA_M
                    + thread_m
                )
                var m_logical = m + m_global_base
                comptime for n_mma in range(num_n_mmas):
                    comptime n_quad, n_within = divmod(n_mma, quad_n_mmas)
                    var n_global = (
                        n
                        + n_quad * half_BN
                        + Int(warp_id_n) * mma_tile_n
                        + n_within * MMA_N
                        + lane_group * c_frag_size
                    )
                    var elem_off = Int32(
                        m_logical * _source_row_stride + n_global
                    )
                    prefetched[m_mma * num_n_mmas + n_mma] = src_bc.load[
                        Self.c_type, c_frag_size
                    ](elem_off)

        _emit_framework_body()

        var c_writer = RegTileEpilogue[
            Self.c_type,
            c_frag_size,
            elementwise_lambda_fn=Self.elementwise_lambda_fn,
        ](c)

        # Drain prefetched loads before the first `v_pk_fma_f32` reads
        # the prefetched array. By now the main loop's MMAs have given
        # the HBM pipeline ample time to complete the prefetches; the
        # wait should be near-zero on memory-bound shapes.
        comptime if has_residual:
            s_waitcnt[vmcnt=UInt32(0)]()

        if m < M and n < N:
            var c_reg = mma_op.accum_tile()

            # FMA (when has_residual) + store.
            comptime for m_mma in range(num_m_mmas):
                comptime m_quad, m_within = divmod(m_mma, quad_m_mmas)
                var m_global_base = (
                    m_quad * half_BM
                    + Int(warp_id_m) * mma_tile_m
                    + m_within * MMA_M
                    + thread_m
                )
                var m_dram = split_id * M + pid_m * BM + m_global_base
                var m_logical = m + m_global_base
                if m_logical < M:
                    comptime for n_mma in range(num_n_mmas):
                        comptime n_quad, n_within = divmod(n_mma, quad_n_mmas)
                        var n_global = (
                            n
                            + n_quad * half_BN
                            + Int(warp_id_n) * mma_tile_n
                            + n_within * MMA_N
                            + lane_group * c_frag_size
                        )
                        var v = (
                            c_reg.tile[1, c_frag_size](m_mma, n_mma)
                            .raw_load[width=c_frag_size](0)
                            .cast[Self.c_type]()
                        )
                        # Pre-residual fused compute lambda — matches
                        # SM100 `D = lambda(Conv(A,B)) + beta * C`
                        # semantics. Fires on the post-cast `c_type`
                        # MMA output (use cases: bias / ReLU / SiLU /
                        # GELU before the skip add).
                        comptime if elementwise_compute_lambda_fn:
                            comptime _compute_fn = (
                                elementwise_compute_lambda_fn.value()
                            )
                            v = _compute_fn[
                                alignment=align_of[
                                    SIMD[Self.c_type, c_frag_size]
                                ]()
                            ](IndexList[2](m_logical, n_global), v)
                        comptime if has_residual:
                            var skip = prefetched[
                                m_mma * num_n_mmas + n_mma
                            ].cast[.float32]()
                            var fused_f32 = v.cast[.float32]() + beta * skip
                            v = fused_f32.cast[Self.c_type]()
                        c_writer.store(v, m=m_dram, n=n_global)


@always_inline
def structured_4wave_matmul[
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
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
) raises:
    """Canonical 4-wave matmul launcher (mirror of `amd_ping_pong_matmul`).

    Single black-box entry point for all dtypes (FP8 + BF16 + FP16) and
    all matmul shape regimes. Internally dispatches on (dtype, M) to
    pick the right (BM, BN, BK, MMA shape) config, and always routes
    through the framework-scheduled body of `AMD4WaveMatmul.run`.

    Production callers go through a higher-level dispatcher (e.g.
    `AMDMatmul`) that knows the shape, dtype, and which other kernels
    are available; it should set `block_{m,n,k}_override` explicitly
    based on its own policy. The internal auto-pick below is a
    convenience default for direct/ad-hoc/benchmark callers; do not
    rely on it from a production dispatcher.

    Recommended tile shapes (measured on MI355X, bf16 N=K=8192):

      M = 1         : use a dedicated GEMV kernel (`linalg/gemv.mojo`),
                      not 4-wave; single-row matvec has its own
                      hardware-aligned dispatch.
      2 ≤ M ≤  64   : use `amd_4wave_split_k_matmul` with
                      `num_splits=4` and BK=128 (the plain kernel
                      under-fills the GPU with too few M-blocks).
      64 < M ≤ 512  : BM=128, BN=128, BK=128  (~1.2× hipBLASLt at
                      M=128; within ~15% at M=512).
      M ≥ 1024      : BM=128, BN=128, BK=64   (BK=128 register-
                      pressure-limits ILP at large M; BK=64 leaves
                      ~88 VGPRs of scheduling headroom).

    These were chosen on raw matmul throughput. The dispatcher may
    legitimately route through 4-wave even when raw matmul is a bit
    slower than a non-fusable vendor BLAS, because
    `elementwise_lambda_fn` can fuse epilogues (bias, GELU, scale,
    residual) and save a separate elementwise kernel launch.

    FP8 keeps the original auto-pick (BM=64 for M ≤ 512, BM=128
    otherwise) and `BK=128` / `mma_shape=(16,16,128)`.

    Parameters:
        a_type: Element type of `a`.
        b_type: Element type of `b`.
        c_type: Element type of `c`.
        enable_swizzle: Enable LDS bank-conflict avoidance.
        block_m_override: If > 0, force BM to this value (must be 64
            or 128). Set this and `block_n_override` together to pin
            the tile shape from a dispatcher.
        block_n_override: If > 0, force BN to this value (must be 64,
            128, or 256). Default 0 uses BM=BN.
        block_k_override: If > 0, force BK to this value. Must be a
            multiple of MMA_K (32 for bf16/fp16, 128 for FP8) and must
            satisfy K % (2*BK) == 0. Valid: 32, 64, 128 for bf16/fp16;
            128 for FP8.
        dump_asm_path: If non-empty, dumps the compiled GCN assembly
            to the given file path. Only used for ASM-level
            diff-debugging.
        elementwise_lambda_fn: Optional fused epilogue. When set, the
            kernel's output store calls the lambda with global
            `(m_global, n_global)` coords and a SIMD-of-`c_frag_size`
            instead of writing to `c` directly.

    Args:
        a: Input tile-tensor for A.
        b: Input tile-tensor for B.
        c: Output tile-tensor for C.
        ctx: Device context used to enqueue the kernel.

    Raises:
        An error if device enqueue fails.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type.is_float8() or a_type == .bfloat16 or a_type == .float16
    ), "4-wave supports float8_e4m3fn, bfloat16, or float16"

    # MMA K-dim selection: FP8 uses MFMA 16x16x128; bf16/fp16 use MFMA
    # 16x16x32. The scheduled MMA op fires `num_k_mmas = BK/MMA_K`
    # MFMAs along K internally via `QuadrantMmaOp`, so BK > MMA_K is
    # transparent to the schedule.
    #
    # SMEM footprint = 2*(BM+BN)*BK*elem_bytes. MI355X has 160 KB LDS.
    # bf16 BM=BN=128, BK=128: 128 KB (fits with headroom for register
    # allocation).
    comptime _is_fp8 = a_type.is_float8()
    comptime _mma_k = 128 if _is_fp8 else 32
    # Default BK=128 for both FP8 and bf16/fp16 (different `num_k_mmas`
    # but same K-tile size). For bf16/fp16 large square (M ≥ 1024) the
    # dispatcher should set `block_k_override=64`.
    comptime _bk = block_k_override if block_k_override > 0 else 128
    # block_k_override validation (BK=0 falls through to the default).
    comptime assert block_k_override == 0 or (
        block_k_override % _mma_k == 0 and block_k_override in (32, 64, 128)
    ), (
        "block_k_override must be 0 (auto) or a multiple of MMA_K in"
        " {32, 64, 128}"
    )

    var N = Int(c.dim[1]())
    var M = Int(c.dim[0]())

    @always_inline
    @__parameter
    def run_kernel[config: MatmulKernelConfig]() raises:
        comptime kernel = AMD4WaveMatmul[
            a_type,
            b_type,
            c_type,
            config,
            enable_swizzle,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ].run[
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(c).LayoutType,
            type_of(a).Engine,
            type_of(b).Engine,
            type_of(c).Engine,
        ]

        var num_blocks_n = ceildiv(N, config.block_shape[1])

        var num_blocks_m = ceildiv(M, config.block_shape[0])
        comptime if dump_asm_path != "":
            ctx.enqueue_function[kernel, dump_asm=dump_asm_path](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )
        else:
            ctx.enqueue_function[kernel](
                a,
                b,
                c,
                grid_dim=(num_blocks_n * num_blocks_m,),
                block_dim=config.num_threads(),
            )

    comptime if block_m_override > 0 and block_n_override > 0:
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

    # Convenience defaults — production dispatchers should set the
    # overrides above. FP8 uses the N=K=4096-tuned cutoff (BM=64 below
    # M=512, BM=128 above). bf16/fp16 default to BM=BN=128 with the
    # comptime-defaulted BK=128, which dominates M ≤ 512 (the prefill
    # regime). For M ≥ 1024 the dispatcher should pass
    # `block_m_override=128, block_n_override=128, block_k_override=64`.
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
