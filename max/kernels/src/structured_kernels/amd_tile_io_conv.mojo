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
"""DRAM->LDS DMA loader for AMD implicit-GEMM convolution.

`TileLoaderLDSIm2col` is a sibling expert of
`structured_kernels.amd_tile_io.TileLoaderLDS` under the same
DRAM→LDS warp-cooperative direction. The split parallels the
existing `TileLoaderLDS` (linear 2D source, coord-indexed) vs
`SubTileLoaderLDS` (single sub-tile, TileTensor-indexed, async-copies
alias scope): same direction, different cooperation or source
shape. Here the differentiator is the **source layout type**: this
loader accepts a 4D NHWC input tensor and `(m_offset, k_offset)`
coordinates in GEMM space (where `M = N*H_out*W_out` and
`K = R*S*C`), translating them to NHWC linear offsets at load time.
The per-lane vs uniform split of each address is preserved so each
iteration still issues a single `buffer_load_*_lds` per lane,
matching the vmcnt accounting `amd_4wave_matmul`'s schedule relies on.

For each lane each iteration, `load_tile` computes

    (n, h_out, w_out) = decompose(m_offset + lane_offset, H_out, W_out)
    (kh, kw, c)       = decompose(k_offset + lane_offset, R, S, C)
    h_in              = h_out * stride_h + kh * dilation_h - pad_h
    w_in              = w_out * stride_w + kw * dilation_w - pad_w
    addr              = ((n*H + h_in)*W + w_in)*C + c

and emits one `buffer_load_*_lds` per lane targeting `addr`. Three
comptime sub-paths inside the body, picked at struct instantiation:

  - **Pure-pointwise fast path** (R=S=1, stride=1, dilation=1,
    pad=0): math collapses to `m*C + k`, identical to
    `TileLoaderLDS(stride = C)`. Same instruction stream as the
    matmul loader; same VGPR usage.
  - **Uniform-substrip path** (general R×S, `tile_cols ≤ C` and
    `C % tile_cols == 0`): each `load_tile` call's lanes all sit in
    one `(kh, kw)` substrip; `(kh, kw, c_base)` are computed once
    per call from `k_offset`. Address-decomp adds the per-lane
    (n, h_out, w_out) divmods.
  - **Per-lane substrip path** (BK > C or non-aligned, e.g. ResNet
    stem with C_in = 64 and BK = 128): each lane independently
    decomposes its `k_lane = k_offset + thread_col`. Adds two more
    divmods per lane but unblocks shapes the fast path can't cover.

Halo (pad > 0) is handled by routing OOB lanes (h_in or w_in outside
`[0, H)`/`[0, W)`) to the SRD's `num_records` sentinel: the AMD
buffer-resource hardware bound check then clamps the read to 0.
Gated by a `_needs_halo_mask` comptime flag so pad=0 callers keep
their exact instruction stream.

The SRD covers the **full NHWC tensor** (`N*H*W*C` elements), not a
per-block slice, so OOB reads (deliberate halo or otherwise) are
bounded by the actual allocation, in contrast to `TileLoaderLDS`
which sizes the SRD to the per-block `.tile[BM, K]()` view. The
full-tensor pattern caps at 4 GiB (AMDBufferResource.num_records is
a 32-bit byte count); fine for typical NHWC conv inputs but watch
the cap when N*H*W*C grows.

K-padding is the caller's responsibility (the 4-wave schedule
requires `K_filter % (2*BK) == 0`). Set `C` to the real input
channel count and pre-pad the filter's trailing K columns with
zeros; the address math naturally handles the padded K reads via
SRD-OOB or filter-zero multiplies in the MMA.
"""

from std.math import ceildiv, min
from std.sys import size_of, simd_width_of
from max.gpu import WARP_SIZE
from max.gpu.intrinsics import AMDBufferResource
from std.memory import AddressSpace
from std.sys.intrinsics import readfirstlane
from std.collections import Optional

from layout import TensorEngine, TensorLayout, TileTensor
from layout.swizzle import Swizzle

from structured_kernels.amd_tile_io import GMemTile, SMemTile, TileLoader


# ===----------------------------------------------------------------------=== #
# TileLoaderLDSIm2col: NHWC -> LDS DMA expert for implicit-GEMM conv
# ===----------------------------------------------------------------------=== #


struct TileLoaderLDSIm2col[
    dtype_: DType,
    tile_rows_: Int,
    tile_cols_: Int,
    C: Int,
    num_loading_warps: Int,
    # Conv input/output spatial geometry (comptime — typical conv layers
    # have static H/W per layer, so per-shape specialization is fine and
    # lets the M decomposition constant-fold the divisor in
    # `divmod(m, H_out*W_out)` and `divmod(m_within, W_out)` to
    # multiply-by-magic-number lowering).
    H: Int = 1,
    W: Int = 1,
    H_out: Int = 1,
    W_out: Int = 1,
    # Filter / conv geometry (comptime).
    R: Int = 1,
    S: Int = 1,
    stride_h: Int = 1,
    stride_w: Int = 1,
    dilation_h: Int = 1,
    dilation_w: Int = 1,
    pad_h: Int = 0,
    pad_w: Int = 0,
    # Optional 3D (NDHWC + Q×R×S filter) mode. Defaults reproduce the
    # 2D loader exactly so existing 2D callers compile unchanged. When
    # Q > 1, the loader expects a 5D NDHWC input and decomposes
    # M = N*D_out*H_out*W_out, K = Q*R*S*C (vs 2D: M = N*H_out*W_out,
    # K = R*S*C). Halo bounds extend to D when pad_d > 0.
    Q: Int = 1,
    D: Int = 1,
    D_out: Int = 1,
    stride_d: Int = 1,
    dilation_d: Int = 1,
    pad_d: Int = 0,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    load_width: Int = simd_width_of[dtype_](),
    use_full_tile_width: Bool = False,
    # Runtime-HW path: when True, H/W/H_out/W_out come from runtime
    # constructor args, not the comptime template params above. Used
    # for graph-compiled callers with dynamic image resolution (e.g.
    # FLUX VAE). The K-decomp (`(kh, kw, c)`) and conv params (R, S,
    # stride, dilation, pad) stay comptime so the per-(kh, kw)
    # specialization still applies. Cost: replaces the magic-multiply
    # lowering of `divmod(m, H_out*W_out)` and `divmod(_, W_out)` with
    # runtime divisions. For BF16's non-swizzle path the divmod fires
    # only once per `load_tile` call (carry-incremented per iter); for
    # the FP8 swizzle path it fires once per (lane, iter).
    use_runtime_hw: Bool = False,
](TileLoader):
    """DRAM->LDS DMA expert for implicit-GEMM convolution, NHWC inputs.

    Sibling of `TileLoaderLDS` (linear GEMM source). Each iteration
    issues one `buffer_load_*_lds` per lane, same vmcnt cost as the
    matmul loader. The kernel's K-loop iterates flat `k_offset ∈ [0, R*S*C)`
    in steps of `tile_cols`; the loader internally decomposes
    `k_offset → (kh, kw, c_offset)` and per-lane `m_lane → (n, h_out, w_out)`,
    then computes `addr = ((n*H + h_in)*W + w_in)*C + c` for each lane.

    The body picks one of three comptime sub-paths at instantiation:
    pure-pointwise (R=S=1, no pad: math collapses to `m*C + k`);
    uniform-substrip (general R×S with `tile_cols ≤ C` and
    `C % tile_cols == 0`: one (kh, kw) per call); per-lane substrip
    (otherwise: each lane decomposes its own `k_lane`). Pad > 0
    additionally routes halo lanes (h_in or w_in outside `[0, H)`/`[0, W)`)
    to the SRD-OOB sentinel.

    Parameters:
        dtype_: Element data type. Re-bound to `dtype` at body scope to
            match the `TileLoader` trait alias.
        tile_rows_: Height of each half-tile to load (in M = N*H_out*W_out
            space). Re-bound to `tile_rows` at body scope.
        tile_cols_: Width of each half-tile (in K = R*S*C space). Re-bound
            to `tile_cols` at body scope. Must satisfy `tile_cols ≤ C`
            and `C % tile_cols == 0` so each `load_tile` call lives
            inside one (kh, kw) substrip.
        C: Input channel count.
        num_loading_warps: Warps cooperating on each load.
        H: Input spatial height.
        W: Input spatial width.
        H_out: Output spatial height (with stride=1, dilation=1, no pad:
            `H - R + 1`).
        W_out: Output spatial width.
        R: Filter height.
        S: Filter width.
        stride_h: Vertical conv stride (>= 1).
        stride_w: Horizontal conv stride (>= 1).
        dilation_h: Vertical conv dilation (>= 1).
        dilation_w: Horizontal conv dilation (>= 1).
        pad_h: Vertical pad (>= 0). Halo lanes route to the SRD-OOB
            sentinel when pad > 0.
        pad_w: Horizontal pad (>= 0).
        Q: Filter temporal extent (3D-only). `Q == 1` (default) keeps
            the loader in 2D mode (4D NHWC input). `Q > 1` activates
            3D mode (5D NDHWC input, K = Q*R*S*C).
        D: Input temporal depth (3D-only; unused when Q == 1).
        D_out: Output temporal depth (3D-only).
        stride_d: Temporal conv stride (3D-only, >= 1).
        dilation_d: Temporal conv dilation (3D-only, >= 1).
        pad_d: Temporal pad (3D-only, >= 0). Halo lanes route to the
            SRD-OOB sentinel when pad_d > 0.
        swizzle: Optional byte-space swizzle for LDS bank conflicts.
        load_width: Elements per load (SIMD width).
        use_full_tile_width: FP8 row-major mode (matches
            `TileLoaderLDS.use_full_tile_width`).
        use_runtime_hw: When True, H/W/H_out/W_out (and D/D_out in 3D
            mode) come from runtime constructor args instead of the
            comptime template params above. Used for graph-compiled
            callers with dynamic image resolution (e.g. FLUX VAE). The
            K-decomposition and conv params (Q, R, S, stride, dilation,
            pad) stay comptime.
    """

    # Body-level aliases re-bind the parametric `dtype_`/`tile_rows_`/
    # `tile_cols_` (trailing-underscore to avoid the struct-param vs
    # trait-alias name clash) to the names the `TileLoader` trait
    # declares. Lets the conformance check match, and lets the rest of
    # this struct keep its `Self.dtype` / `Self.tile_*` references
    # unchanged.
    comptime dtype: DType = Self.dtype_
    comptime tile_rows: Int = Self.tile_rows_
    comptime tile_cols: Int = Self.tile_cols_

    # General geometry guard rails. The address-math expression
    # `h_in = h_out * stride_h + kh * dilation_h - pad_h` (and the W
    # analogue) handles arbitrary positive stride and dilation; the
    # SRD-OOB sentinel handles arbitrary pad >= 0. The earlier
    # stride=1 / dilation=1 restrictions have been lifted.
    comptime _geom_ok = (
        Self.R >= 1
        and Self.S >= 1
        and Self.stride_h >= 1
        and Self.stride_w >= 1
        and Self.dilation_h >= 1
        and Self.dilation_w >= 1
        and Self.pad_h >= 0
        and Self.pad_w >= 0
        and Self.Q >= 1
        and Self.D >= 1
        and Self.D_out >= 1
        and Self.stride_d >= 1
        and Self.dilation_d >= 1
        and Self.pad_d >= 0
    )
    # 3D-conv mode: present (5D NDHWC input + Q×R×S filter) when Q > 1.
    # Q == 1 is the 2D default (4D NHWC input). Other 3D-only params
    # (D, D_out, stride_d, dilation_d, pad_d) are ignored when Q == 1
    # so 2D callers don't have to set them.
    comptime _is_3d = Self.Q > 1
    # Pure-pointwise specialization (M1 fast path): uses the linear
    # `m*C + k` address math without M-decomposition. Requires no pad
    # because pad > 0 would create OOB conditions the fast path can't
    # mask cheaply. 3D mode is never pure-pointwise (the Q axis adds
    # the d_in computation even if Q=1=R=S, but we already gate _is_3d
    # on Q>1).
    comptime _is_pure_pointwise = (
        Self.R == 1
        and Self.S == 1
        and Self.stride_h == 1
        and Self.stride_w == 1
        and Self.dilation_h == 1
        and Self.dilation_w == 1
        and Self.pad_h == 0
        and Self.pad_w == 0
        and not Self._is_3d
    )
    # When pad>0, halo lanes need to route to the SRD-OOB sentinel.
    # Gated so M2 (interior-only) keeps its exact instruction stream.
    comptime _needs_halo_mask = (
        Self.pad_h > 0 or Self.pad_w > 0 or Self.pad_d > 0
    )

    # When `tile_cols <= C and C % tile_cols == 0`, each load_tile call
    # stays inside one (kh, kw) substrip → fast path with uniform substrip
    # per call. Otherwise the slow path computes per-lane (kh, kw, c).
    # The fast-path condition is what M1/M2/M3 tests with C_in ≥ BK and
    # C_in % BK == 0 hit; the slow path covers C_in < BK (e.g. ResNet's
    # first conv C_in=64) and C_in not aligned to BK.
    comptime _is_uniform_substrip = (
        Self.tile_cols <= Self.C and Self.C % Self.tile_cols == 0
    )

    # Derivations mirror `TileLoaderLDS`. The `stride` between adjacent
    # M-elements in NHWC linear memory is C (W*C jumps occur across
    # W-axis boundaries; for milestone 1, packed NHWC means within a
    # contiguous M-block the stride is uniformly C).
    comptime subtile_cols = Self.tile_cols if Self.use_full_tile_width else 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width
    comptime thread_rows = WARP_SIZE // Self.threads_per_row

    comptime num_warp_cols = Self.tile_cols // Self.subtile_cols
    comptime num_warp_rows = Self.num_loading_warps // Self.num_warp_cols

    comptime elements_per_warp = WARP_SIZE * Self.load_width
    comptime rows_per_warp = Self.elements_per_warp // Self.tile_cols

    comptime loading_threads = Self.num_loading_warps * WARP_SIZE
    comptime rows_per_iteration = Self.loading_threads // (
        Self.tile_cols // Self.load_width
    )
    # `ceildiv` (not floor-div) so that under-supplied sub-tiles —
    # `tile_rows < rows_per_iteration` (e.g. half_BM=32 + BK=32 + bf16
    # where 256 loading threads can transfer the whole 32x32 sub-tile in
    # ~half a wave) — get `num_iterations == 1`. Floor-div rounded this
    # to 0, the `comptime for i in range(0)` unrolled 0 times, and the
    # loader silently emitted zero `buffer_load_lds` — LDS uninit, MMA
    # garbage. Matches `TileLoaderLDS`. The `load_tile` body gates
    # over-supplied warps via `warp_id < active_warps_this_iter`.
    comptime num_iterations = ceildiv(Self.tile_rows, Self.rows_per_iteration)
    comptime total_warp_rows = ceildiv(Self.tile_rows, Self.rows_per_warp)

    comptime warp_subtile_bytes = Self.rows_per_warp * Self.tile_cols * size_of[
        Self.dtype
    ]()
    comptime lane_load_bytes = Self.load_width * size_of[Self.dtype]()
    comptime row_bytes = Self.tile_cols * size_of[Self.dtype]()

    comptime _needs_per_iter_swizzle = Bool(
        Self.swizzle
    ) and Self.use_full_tile_width

    var buffer: AMDBufferResource
    var thread_row: Int
    var thread_col: Int
    var warp_id: Int
    var lane_id: Int
    # Stored only when the halo-mask path is active (pad>0). Used as
    # the OOB sentinel: setting a lane's vector_offset to this value
    # routes the SRD to its zero-clamp behavior. A single SGPR-class
    # field; zero VGPR cost on the M2 interior path because the field
    # access is guarded by `comptime if Self._needs_halo_mask`.
    var num_records: Int
    # Block anchor in (M, K) GEMM space. Caller-supplied at construction.
    # `load_tile(m_offset, k_offset)` addresses `(m_anchor + m_offset,
    # k_anchor + k_offset)` in flat GEMM coords (where M = N*H_out*W_out
    # and K = R*S*C). Matches the matmul `TileLoaderLDS` sibling so the
    # kernel callsite can pass within-block offsets uniformly.
    var m_anchor: Int
    var k_anchor: Int
    # Runtime conv geometry — populated when `use_runtime_hw=True`. The
    # comptime path stores zeros (negligible: per-block constants, the
    # AMD lowering pins them to SGPRs via `readfirstlane`). Pre-computed
    # spatial = h * w_out * h_out caches one multiply per `load_tile`.
    var rt_h: Int
    var rt_w: Int
    var rt_h_out: Int
    var rt_w_out: Int
    var rt_spatial: Int  # h_out * w_out
    # 3D-only runtime conv geometry. Unused (zeroed) when `_is_3d=False`
    # or on the static 3D path; same per-block-constant SGPR cost as
    # the 2D rt_* fields.
    var rt_d: Int
    var rt_d_out: Int
    var rt_spatial_dhw: Int  # d_out * h_out * w_out

    @staticmethod
    def _validate_geometry():
        """Tier 2/3 sanity asserts on the loader's derived counts.

        The class body computes a chain of integer divisions
        (`subtile_cols // load_width`, `tile_cols // subtile_cols`,
        ..., `tile_rows // rows_per_iteration`). When any link floors
        to 0, downstream `comptime for` loops unroll 0 times: loader
        emits no `buffer_load_lds` at all, LDS stays uninitialized,
        MMA reads garbage. These asserts make every link's invariant
        explicit. Mirrors `TileLoaderLDS.__init__`'s checks.
        """
        comptime assert Self.threads_per_row >= 1, (
            "threads_per_row = subtile_cols // load_width must be >= 1"
            " (subtile_cols >= load_width)."
        )
        comptime assert (
            Self.subtile_cols % Self.load_width == 0
        ), "subtile_cols must be a multiple of load_width."
        comptime assert Self.num_warp_cols >= 1, (
            "num_warp_cols = tile_cols // subtile_cols must be >= 1"
            " (tile_cols >= subtile_cols)."
        )
        comptime assert (
            Self.tile_cols % Self.subtile_cols == 0
        ), "tile_cols must be a multiple of subtile_cols."
        comptime assert Self.num_loading_warps % Self.num_warp_cols == 0, (
            "num_loading_warps must be a multiple of num_warp_cols"
            " (otherwise num_warp_rows loses warps)."
        )
        comptime assert Self.rows_per_warp >= 1, (
            "rows_per_warp = WARP_SIZE * load_width // tile_cols must be"
            " >= 1 (each warp must cover at least one row)."
        )
        comptime assert Self.rows_per_iteration >= 1, (
            "rows_per_iteration must be >= 1 (loading_threads /"
            " (tile_cols / load_width)). Sub-tile too narrow for"
            " 4-wave coverage."
        )
        comptime assert Self.num_iterations >= 1, (
            "num_iterations = ceildiv(tile_rows, rows_per_iteration) == 0"
            " — tile_rows must be >= 1."
        )
        comptime assert Self.total_warp_rows >= 1, (
            "total_warp_rows = ceildiv(tile_rows, rows_per_warp) == 0"
            " — tile_rows must be >= 1."
        )

    @always_inline
    def __init__[
        InLayout: TensorLayout,
        InEngine: TensorEngine,
    ](
        out self,
        src_nhwc: TileTensor[
            mut=False, Self.dtype, InLayout, _, Engine=InEngine
        ],
        warp_id: Int,
        lane_id: Int,
        *,
        m_anchor: Int = 0,
        k_anchor: Int = 0,
    ):
        """Builds the loader from a 4D NHWC input TileTensor.

        The SRD covers the entire NHWC tensor (`N*H*W*C` elements).
        Per-block addressing is split between `m_anchor`/`k_anchor`
        (per-block origin in GEMM space, set at construction) and the
        `m_offset`/`k_offset` args of `load_tile` (within-block).

        This overload is for the **comptime-HW path** (the default);
        the runtime conv geometry fields are populated with zeros and
        the loader uses the comptime template params instead.

        Parameters:
            InLayout: `TensorLayout` of the input `src_nhwc` `TileTensor`.
                Must be rank-4 NHWC (or rank-5 NDHWC when `Q > 1`) with
                static spatial dims matching the struct's `H`, `W`, and
                `C` params.
            InEngine: `TensorEngine` of the input `src_nhwc` `TileTensor`.

        Args:
            src_nhwc: 4D NHWC input tensor of shape `(N, H, W, C)`.
            warp_id: Warp identifier within the loading warp group.
            lane_id: Lane identifier within the warp.
            m_anchor: M-coordinate (= flat N*H_out*W_out index) of the
                block origin. Added to `m_offset` at load time.
                Defaults to 0: pass per-block origin from the kernel.
            k_anchor: K-coordinate (= flat (kh, kw, c) index) of the
                block origin. Added to `k_offset` at load time.
                Defaults to 0: conv split-K is not yet supported, so
                callers typically leave this at the default.
        """
        Self._validate_geometry()
        comptime assert not Self.use_runtime_hw, (
            "use_runtime_hw=True requires the (h, w, h_out, w_out)"
            " constructor overload."
        )
        comptime assert Self._geom_ok, (
            "TileLoaderLDSIm2col requires R>=1, S>=1, Q>=1, stride>=1,"
            " dilation>=1, pad>=0."
        )
        comptime if Self._is_3d:
            comptime assert (
                InLayout.rank == 5
            ), "TileLoaderLDSIm2col (Q>1) expects a rank-5 NDHWC TileTensor."
            comptime assert InLayout.static_shape[4] == Self.C, (
                "TileLoaderLDSIm2col.C parameter must match the input's"
                " static C dim (NDHWC.shape[4])."
            )
        else:
            comptime assert (
                InLayout.rank == 4
            ), "TileLoaderLDSIm2col (Q==1) expects a rank-4 NHWC TileTensor."
            comptime assert InLayout.static_shape[3] == Self.C, (
                "TileLoaderLDSIm2col.C parameter must match the input's"
                " static C dim (NHWC.shape[3])."
            )
        # When `tile_cols <= C and C % tile_cols == 0`, each load_tile call
        # stays within a single (kh, kw) substrip — `load_tile`'s general
        # path takes the uniform-substrip fast path (kh, kw, c_base
        # computed once per call). Otherwise (BK > C, or C not a multiple
        # of BK), the per-lane k_lane spans multiple substrips, and the
        # slow path computes (kh_lane, kw_lane, c_lane) per-lane per call.
        # No comptime assert here — both paths are correct; the comptime
        # branch in load_tile picks the right one.
        # For non-pointwise cases (R > 1 or S > 1) the conv geometry
        # parameters must be set explicitly — defaults of 1 are only
        # valid for the 1×1 fast path where the math collapses.
        comptime if not Self._is_pure_pointwise:
            # Effective receptive-field extents accounting for dilation.
            comptime _eff_R = Self.dilation_h * (Self.R - 1) + 1
            comptime _eff_S = Self.dilation_w * (Self.S - 1) + 1
            comptime assert (
                Self.H + 2 * Self.pad_h >= _eff_R
            ), "H + 2*pad_h must be >= dilation_h*(R-1) + 1"
            comptime assert (
                Self.W + 2 * Self.pad_w >= _eff_S
            ), "W + 2*pad_w must be >= dilation_w*(S-1) + 1"
            comptime assert (
                Self.H_out
                == (Self.H + 2 * Self.pad_h - _eff_R) // Self.stride_h + 1
            ), (
                "H_out must equal (H + 2*pad_h - dilation_h*(R-1) - 1)"
                " // stride_h + 1"
            )
            comptime assert (
                Self.W_out
                == (Self.W + 2 * Self.pad_w - _eff_S) // Self.stride_w + 1
            ), (
                "W_out must equal (W + 2*pad_w - dilation_w*(S-1) - 1)"
                " // stride_w + 1"
            )
            comptime if Self._is_3d:
                # NDHWC: dim[1]=D, dim[2]=H, dim[3]=W.
                comptime _eff_Q = Self.dilation_d * (Self.Q - 1) + 1
                comptime assert (
                    Self.D + 2 * Self.pad_d >= _eff_Q
                ), "D + 2*pad_d must be >= dilation_d*(Q-1) + 1"
                comptime assert (
                    Self.D_out
                    == (Self.D + 2 * Self.pad_d - _eff_Q) // Self.stride_d + 1
                ), (
                    "D_out must equal (D + 2*pad_d - dilation_d*(Q-1) - 1)"
                    " // stride_d + 1"
                )
                comptime assert (
                    InLayout.static_shape[1] == Self.D
                ), "Loader D param must match input.shape[1] (NDHWC)"
                comptime assert (
                    InLayout.static_shape[2] == Self.H
                ), "Loader H param must match input.shape[2] (NDHWC)"
                comptime assert (
                    InLayout.static_shape[3] == Self.W
                ), "Loader W param must match input.shape[3] (NDHWC)"
            else:
                comptime assert (
                    InLayout.static_shape[1] == Self.H
                ), "Loader H param must match input.shape[1]"
                comptime assert (
                    InLayout.static_shape[2] == Self.W
                ), "Loader W param must match input.shape[2]"

        # SRD over the whole input buffer. `num_records` is in elements;
        # the descriptor multiplies by `size_of[dtype]` internally.
        comptime assert Self.C > 0, "TileLoaderLDSIm2col.C must be > 0"
        var nr: Int
        comptime if Self._is_3d:
            # NDHWC: 5D.
            var n_5d = Int(src_nhwc.dim[0]())
            var d_5d = Int(src_nhwc.dim[1]())
            var h_5d = Int(src_nhwc.dim[2]())
            var w_5d = Int(src_nhwc.dim[3]())
            var c_5d = Int(src_nhwc.dim[4]())
            nr = n_5d * d_5d * h_5d * w_5d * c_5d
        else:
            # NHWC: 4D.
            var n = Int(src_nhwc.dim[0]())
            var h = Int(src_nhwc.dim[1]())
            var w = Int(src_nhwc.dim[2]())
            var c = Int(src_nhwc.dim[3]())
            nr = n * h * w * c
        self.buffer = AMDBufferResource(
            readfirstlane(src_nhwc.ptr), readfirstlane(nr)
        )
        self.num_records = nr

        self.warp_id = warp_id
        self.lane_id = lane_id

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

        self.m_anchor = m_anchor
        self.k_anchor = k_anchor

        # Runtime-HW fields are unused on the comptime path; zero them
        # so the struct's TrivialRegisterPassable invariants hold. The
        # AMD lowering pins these per-block constants to SGPRs.
        self.rt_h = 0
        self.rt_w = 0
        self.rt_h_out = 0
        self.rt_w_out = 0
        self.rt_spatial = 0
        self.rt_d = 0
        self.rt_d_out = 0
        self.rt_spatial_dhw = 0

    @always_inline
    def __init__[
        InLayout: TensorLayout,
        InEngine: TensorEngine,
    ](
        out self,
        src_nhwc: TileTensor[
            mut=False, Self.dtype, InLayout, _, Engine=InEngine
        ],
        warp_id: Int,
        lane_id: Int,
        *,
        runtime_h: Int,
        runtime_w: Int,
        runtime_h_out: Int,
        runtime_w_out: Int,
        m_anchor: Int = 0,
        k_anchor: Int = 0,
    ):
        """Runtime-HW overload: H/W/H_out/W_out from runtime args.

        Equivalent to the comptime-HW overload except the conv input
        / output spatial dims are runtime values (typically read from
        `input.dim()` / `output.dim()` by the launcher). Use when the
        graph compiler can't pin the resolution.

        Parameters:
            InLayout: `TensorLayout` of the input `src_nhwc` `TileTensor`.
            InEngine: `TensorEngine` of the input `src_nhwc` `TileTensor`.

        Args:
            src_nhwc: 4D NHWC input tensor of shape `(N, H, W, C)`.
            warp_id: Warp identifier within the loading warp group.
            lane_id: Lane identifier within the warp.
            runtime_h: Runtime input height.
            runtime_w: Runtime input width.
            runtime_h_out: Runtime output height (must equal `(runtime_h
                + 2*pad_h - dilation_h*(R-1) - 1) // stride_h + 1`).
            runtime_w_out: Runtime output width.
            m_anchor: M-coordinate of the block origin. Added to
                `m_offset` at load time. Defaults to 0.
            k_anchor: K-coordinate of the block origin. Added to
                `k_offset` at load time. Defaults to 0.
        """
        Self._validate_geometry()
        comptime assert (
            Self.use_runtime_hw
        ), "this constructor requires use_runtime_hw=True"
        comptime assert Self._geom_ok, (
            "TileLoaderLDSIm2col requires R>=1, S>=1, Q>=1, stride>=1,"
            " dilation>=1, pad>=0."
        )
        comptime assert not Self._is_3d, (
            "TileLoaderLDSIm2col (Q>1) runtime-HW requires the 5D"
            " (runtime_d/d_out + ...) constructor overload."
        )
        comptime assert (
            InLayout.rank == 4
        ), "TileLoaderLDSIm2col expects a rank-4 NHWC TileTensor."
        # The static-C consistency check is skipped on the runtime-HW
        # path: callers may pass a fully dynamic layout (e.g. NHWC
        # with all four dims runtime) but the kernel still uses
        # `Self.C` (comptime template param) for all C-relative
        # arithmetic. Only require static_shape[3] == Self.C when the
        # layout carries it; allow -1 (dynamic) otherwise.
        comptime assert (
            InLayout.static_shape[3] == Self.C or InLayout.static_shape[3] == -1
        ), (
            "TileLoaderLDSIm2col.C parameter must match the input's"
            " static C dim (NHWC.shape[3]) when that dim is static."
        )

        # SRD over the whole NHWC buffer. Same shape as the comptime
        # constructor; `nr` is in elements.
        var n = Int(src_nhwc.dim[0]())
        var c = Int(src_nhwc.dim[3]())
        comptime assert Self.C > 0, "TileLoaderLDSIm2col.C must be > 0"
        var nr = n * runtime_h * runtime_w * c
        self.buffer = AMDBufferResource(
            readfirstlane(src_nhwc.ptr), readfirstlane(nr)
        )
        self.num_records = nr

        self.warp_id = warp_id
        self.lane_id = lane_id

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

        self.m_anchor = m_anchor
        self.k_anchor = k_anchor

        # Runtime conv geometry: store + cache spatial = h_out * w_out.
        self.rt_h = runtime_h
        self.rt_w = runtime_w
        self.rt_h_out = runtime_h_out
        self.rt_w_out = runtime_w_out
        self.rt_spatial = runtime_h_out * runtime_w_out
        # 3D-only fields unused on the 2D runtime-HW path.
        self.rt_d = 0
        self.rt_d_out = 0
        self.rt_spatial_dhw = 0

    @always_inline
    def __init__[
        InLayout: TensorLayout,
        InEngine: TensorEngine,
    ](
        out self,
        src_ndhwc: TileTensor[
            mut=False, Self.dtype, InLayout, _, Engine=InEngine
        ],
        warp_id: Int,
        lane_id: Int,
        *,
        runtime_d: Int,
        runtime_h: Int,
        runtime_w: Int,
        runtime_d_out: Int,
        runtime_h_out: Int,
        runtime_w_out: Int,
        m_anchor: Int = 0,
        k_anchor: Int = 0,
    ):
        """3D runtime-HW overload: D/H/W/D_out/H_out/W_out from runtime args.

        Equivalent to the 2D runtime-HW overload but for `Q > 1` mode:
        accepts a rank-5 NDHWC `TileTensor` and runtime D / D_out args
        in addition to the spatial H/W ones. The K-decomposition and
        conv params (Q, R, S, stride_d, stride_h, stride_w, dilation_*,
        pad_*, C) stay comptime.

        Parameters:
            InLayout: `TensorLayout` of the input `src_ndhwc` `TileTensor`.
            InEngine: `TensorEngine` of the input `src_ndhwc` `TileTensor`.

        Args:
            src_ndhwc: 5D NDHWC input tensor of shape `(N, D, H, W, C)`.
            warp_id: Warp identifier within the loading warp group.
            lane_id: Lane identifier within the warp.
            runtime_d: Runtime input depth.
            runtime_h: Runtime input height.
            runtime_w: Runtime input width.
            runtime_d_out: Runtime output depth (must equal
                `(runtime_d + 2*pad_d - dilation_d*(Q-1) - 1) //
                stride_d + 1`).
            runtime_h_out: Runtime output height.
            runtime_w_out: Runtime output width.
            m_anchor: M-coordinate of the block origin in GEMM space
                (`= flat N*D_out*H_out*W_out index`). Added to
                `m_offset` at load time. Defaults to 0.
            k_anchor: K-coordinate of the block origin in GEMM space
                (`= flat Q*R*S*C index`). Added to `k_offset` at load
                time. Defaults to 0.
        """
        Self._validate_geometry()
        comptime assert (
            Self.use_runtime_hw
        ), "this constructor requires use_runtime_hw=True"
        comptime assert Self._geom_ok, (
            "TileLoaderLDSIm2col requires R>=1, S>=1, Q>=1, stride>=1,"
            " dilation>=1, pad>=0."
        )
        comptime assert (
            Self._is_3d
        ), "5D runtime-HW constructor requires Q>1 (3D mode)."
        comptime assert (
            InLayout.rank == 5
        ), "TileLoaderLDSIm2col (Q>1) expects a rank-5 NDHWC TileTensor."
        # Allow dynamic C (-1); enforce match if the layout has a
        # static C dim.
        comptime assert (
            InLayout.static_shape[4] == Self.C or InLayout.static_shape[4] == -1
        ), (
            "TileLoaderLDSIm2col.C parameter must match the input's"
            " static C dim (NDHWC.shape[4]) when that dim is static."
        )

        # SRD over the whole NDHWC buffer. `nr` is in elements.
        var n = Int(src_ndhwc.dim[0]())
        var c = Int(src_ndhwc.dim[4]())
        comptime assert Self.C > 0, "TileLoaderLDSIm2col.C must be > 0"
        var nr = n * runtime_d * runtime_h * runtime_w * c
        self.buffer = AMDBufferResource(
            readfirstlane(src_ndhwc.ptr), readfirstlane(nr)
        )
        self.num_records = nr

        self.warp_id = warp_id
        self.lane_id = lane_id

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

        self.m_anchor = m_anchor
        self.k_anchor = k_anchor

        # 2D runtime fields still get the H/W values (the 3D body uses
        # rt_h/rt_w as the corresponding axes). Cache the 2D and 3D
        # spatial products to avoid per-call multiplies.
        self.rt_h = runtime_h
        self.rt_w = runtime_w
        self.rt_h_out = runtime_h_out
        self.rt_w_out = runtime_w_out
        self.rt_spatial = runtime_h_out * runtime_w_out
        self.rt_d = runtime_d
        self.rt_d_out = runtime_d_out
        self.rt_spatial_dhw = runtime_d_out * runtime_h_out * runtime_w_out

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[Self.dtype, _, _],
        m_offset: Int,
        k_offset: Int,
    ):
        """Loads a half-tile from NHWC global memory into the SMEM dst.

        Two paths:

        - **Pure-pointwise fast path** (R=S=1, stride=1, dilation=1, pad=0):
          GEMM address `addr = ((n*H + h)*W + w)*C + c` collapses to
          `addr = m * C + k`. Identical to `TileLoaderLDS.load_tile` with
          `stride = C`; the per-lane vs uniform offset split is preserved
          so each iteration issues one `buffer_load_*_lds` per lane,
          matching the matmul's vmcnt accounting.

        - **General R×S path** (M2): the lane's `(m_lane, k_lane)` are
          decomposed at runtime: `k_lane → (kh, kw, c)` via comptime R, S,
          C divisors (constant-folded to multiply-by-magic); `m_lane →
          (n, h_out, w_out)` via comptime H_out, W_out divisors. Then
          `h_in = h_out * stride_h + kh * dilation_h - pad_h` (similarly
          for w_in) and `addr = ((n*H + h_in)*W + w_in)*C + c`. The full
          per-lane address goes into `vector_offset`; `scalar_offset = 0`.
          Costs more VGPRs per load than the fast path because the address
          decomposition can't be cleanly split into a uniform + per-lane
          pair (the m → (n, h_out, w_out) decomposition is non-linear).

        Args:
            dst: Destination half-tile in SHARED address space.
            m_offset: M-coordinate within the block (added to
                `self.m_anchor` to form the absolute GEMM M coord =
                flat N*H_out*W_out index).
            k_offset: K-coordinate within the block (added to
                `self.k_anchor` to form the absolute GEMM K coord =
                flat (kh, kw, c) index ∈ [0, R*S*C)). Must be a
                multiple of `tile_cols`.
        """
        comptime SmemPtr = Pointer[
            Scalar[Self.dtype], MutAnyOrigin, address_space=.SHARED
        ]

        # Absolute GEMM coords for this call. Anchors default to 0, so
        # legacy callers that pass absolute offsets keep their address
        # math unchanged.
        var m_eff = self.m_anchor + m_offset
        var k_eff = self.k_anchor + k_offset

        comptime if Self._is_pure_pointwise:
            # -- Pure-pointwise fast path (M1: R=S=1 etc.) --
            comptime if Self._needs_per_iter_swizzle:
                var lane_byte = self.lane_id * Self.lane_load_bytes

                comptime for i in range(Self.num_iterations):
                    # See `total_warp_rows` doc — gate over-supplied
                    # warps when the sub-tile doesn't fill the grid.
                    comptime active_warps_this_iter = min(
                        Self.num_loading_warps,
                        Self.total_warp_rows - i * Self.num_loading_warps,
                    )
                    var tile_idx = i * Self.num_loading_warps + self.warp_id
                    var warp_tile = dst.tile[
                        Self.rows_per_warp, Self.tile_cols
                    ](tile_idx, 0)
                    var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                    var full_byte = (
                        tile_idx * Self.warp_subtile_bytes + lane_byte
                    )
                    var swizzled_byte = Self.swizzle.value()(full_byte)

                    var swizzled_row = swizzled_byte // Self.row_bytes
                    var swizzled_col = (
                        swizzled_byte % Self.row_bytes
                    ) // size_of[Self.dtype]()

                    var lane_offset = swizzled_col + swizzled_row * Self.C
                    var uniform_offset = k_eff + m_eff * Self.C

                    comptime if active_warps_this_iter == Self.num_loading_warps:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(lane_offset),
                            smem_ptr,
                            scalar_offset=Int32(uniform_offset),
                        )
                    else:
                        if Int(self.warp_id) < active_warps_this_iter:
                            self.buffer.load_to_lds[width=Self.load_width](
                                Int32(lane_offset),
                                smem_ptr,
                                scalar_offset=Int32(uniform_offset),
                            )
            else:
                var lane_offset = self.thread_col + self.thread_row * Self.C

                comptime for i in range(Self.num_iterations):
                    comptime active_warps_this_iter = min(
                        Self.num_loading_warps,
                        Self.total_warp_rows - i * Self.num_loading_warps,
                    )
                    var tile_idx = i * Self.num_loading_warps + self.warp_id
                    var warp_tile = dst.tile[
                        Self.rows_per_warp, Self.tile_cols
                    ](tile_idx, 0)
                    var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                    var tile_row = m_eff + i * Self.rows_per_iteration
                    var uniform_offset = k_eff + tile_row * Self.C

                    comptime if active_warps_this_iter == Self.num_loading_warps:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(lane_offset),
                            smem_ptr,
                            scalar_offset=Int32(uniform_offset),
                        )
                    else:
                        if Int(self.warp_id) < active_warps_this_iter:
                            self.buffer.load_to_lds[width=Self.load_width](
                                Int32(lane_offset),
                                smem_ptr,
                                scalar_offset=Int32(uniform_offset),
                            )
        else:
            # -- General R×S path (M2+) --
            # Two comptime-selected sub-paths for the K-decomposition:
            #   - Uniform-substrip (fast): `tile_cols <= C` and
            #     `C % tile_cols == 0`. Each load_tile call's lanes all
            #     fall in a single (kh, kw) substrip; kh/kw/c_base are
            #     computed once per call from k_offset.
            #   - Per-lane substrip (slow): general case (e.g. BK > C
            #     when C_in=64 for FP8 with BK=128). Each lane's
            #     k_lane = k_offset + (thread_col or swizzled_col) may sit
            #     in a different substrip, so (kh, kw, c_lane) are
            #     computed per-lane.
            # Shadow H/W/H_out/W_out so the rest of this path is
            # agnostic to comptime-vs-runtime conv geometry. The
            # comptime branch lets Mojo fold the divmods to magic-
            # multiplies; the runtime branch issues real divisions.
            var H_eff: Int
            var W_eff: Int
            var H_out_eff: Int
            var W_out_eff: Int
            var spatial: Int  # 2D mode: H_out*W_out; 3D mode: H_out*W_out
            # 3D-only effective geometry. Zero in 2D mode (unused, the
            # comptime branches below skip the 3D code path entirely).
            var D_eff: Int
            var D_out_eff: Int
            var spatial_dhw: Int  # 3D only: D_out*H_out*W_out
            comptime if Self.use_runtime_hw:
                H_eff = self.rt_h
                W_eff = self.rt_w
                H_out_eff = self.rt_h_out
                W_out_eff = self.rt_w_out
                spatial = self.rt_spatial
                D_eff = self.rt_d
                D_out_eff = self.rt_d_out
                spatial_dhw = self.rt_spatial_dhw
            else:
                H_eff = Self.H
                W_eff = Self.W
                H_out_eff = Self.H_out
                W_out_eff = Self.W_out
                spatial = Self.H_out * Self.W_out
                D_eff = Self.D
                D_out_eff = Self.D_out
                spatial_dhw = Self.D_out * Self.H_out * Self.W_out

            comptime if Self._needs_per_iter_swizzle:
                var lane_byte = self.lane_id * Self.lane_load_bytes

                # Uniform-substrip values hoisted (used only on the
                # fast path; harmless extra comptime work otherwise).
                # 3D adds kq (filter depth slot). Decomposition is
                # k = kq*R*S*C + kh*S*C + kw*C + c.
                var substrip_uniform = 0
                var kq_uniform = 0
                var kh_uniform = 0
                var kw_uniform = 0
                var c_base_uniform = 0
                comptime if Self._is_uniform_substrip:
                    substrip_uniform = k_eff // Self.C
                    comptime if Self._is_3d:
                        var rs_uniform = substrip_uniform % (Self.R * Self.S)
                        kq_uniform = substrip_uniform // (Self.R * Self.S)
                        kh_uniform = rs_uniform // Self.S
                        kw_uniform = rs_uniform % Self.S
                    else:
                        kh_uniform = substrip_uniform // Self.S
                        kw_uniform = substrip_uniform % Self.S
                    c_base_uniform = k_eff % Self.C

                comptime for i in range(Self.num_iterations):
                    comptime active_warps_this_iter = min(
                        Self.num_loading_warps,
                        Self.total_warp_rows - i * Self.num_loading_warps,
                    )
                    var tile_idx = i * Self.num_loading_warps + self.warp_id
                    var warp_tile = dst.tile[
                        Self.rows_per_warp, Self.tile_cols
                    ](tile_idx, 0)
                    var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                    var full_byte = (
                        tile_idx * Self.warp_subtile_bytes + lane_byte
                    )
                    var swizzled_byte = Self.swizzle.value()(full_byte)

                    var swizzled_row = swizzled_byte // Self.row_bytes
                    var swizzled_col = (
                        swizzled_byte % Self.row_bytes
                    ) // size_of[Self.dtype]()

                    var m_lane = m_eff + swizzled_row

                    # Pick (kq, kh, kw, c_lane) per the comptime
                    # fast/slow choice. kq is only used in 3D mode.
                    var kq: Int = 0
                    var kh: Int
                    var kw: Int
                    var c_lane: Int
                    comptime if Self._is_uniform_substrip:
                        kq = kq_uniform
                        kh = kh_uniform
                        kw = kw_uniform
                        c_lane = c_base_uniform + swizzled_col
                    else:
                        var k_lane = k_eff + swizzled_col
                        var substrip_lane = k_lane // Self.C
                        comptime if Self._is_3d:
                            var rs_lane = substrip_lane % (Self.R * Self.S)
                            kq = substrip_lane // (Self.R * Self.S)
                            kh = rs_lane // Self.S
                            kw = rs_lane % Self.S
                        else:
                            kh = substrip_lane // Self.S
                            kw = substrip_lane % Self.S
                        c_lane = k_lane % Self.C

                    # M decomposition.
                    #   2D: m_lane → (n, h_out, w_out)
                    #   3D: m_lane → (n, d_out, h_out, w_out)
                    var n: Int
                    var d_out: Int = 0  # 2D: unused
                    var h_out: Int
                    var w_out: Int
                    comptime if Self._is_3d:
                        var n_within_dhw = divmod(m_lane, spatial_dhw)
                        n = n_within_dhw[0]
                        var m_within_dhw = n_within_dhw[1]
                        var d_within_hw = divmod(m_within_dhw, spatial)
                        d_out = d_within_hw[0]
                        var m_within_hw = d_within_hw[1]
                        var h_w = divmod(m_within_hw, W_out_eff)
                        h_out = h_w[0]
                        w_out = h_w[1]
                    else:
                        var n_m_within = divmod(m_lane, spatial)
                        n = n_m_within[0]
                        var m_within = n_m_within[1]
                        var h_w = divmod(m_within, W_out_eff)
                        h_out = h_w[0]
                        w_out = h_w[1]

                    var h_in = (
                        h_out * Self.stride_h
                        + kh * Self.dilation_h
                        - Self.pad_h
                    )
                    var w_in = (
                        w_out * Self.stride_w
                        + kw * Self.dilation_w
                        - Self.pad_w
                    )
                    # Address. 2D: ((n*H + h_in)*W + w_in)*C + c.
                    # 3D: (((n*D + d_in)*H + h_in)*W + w_in)*C + c.
                    var addr_safe: Int
                    var d_in: Int = 0  # 2D: unused
                    comptime if Self._is_3d:
                        d_in = (
                            d_out * Self.stride_d
                            + kq * Self.dilation_d
                            - Self.pad_d
                        )
                        addr_safe = (
                            ((n * D_eff + d_in) * H_eff + h_in) * W_eff + w_in
                        ) * Self.C + c_lane
                    else:
                        addr_safe = (
                            (n * H_eff + h_in) * W_eff + w_in
                        ) * Self.C + c_lane
                    var addr = addr_safe
                    comptime if Self._needs_halo_mask:
                        # Halo lanes (any of d_in / h_in / w_in outside
                        # bounds) route to `num_records`; the SRD's
                        # hardware bound check clamps the read to 0.
                        var in_bounds = (
                            (h_in >= 0)
                            and (h_in < H_eff)
                            and (w_in >= 0)
                            and (w_in < W_eff)
                        )
                        comptime if Self._is_3d:
                            in_bounds = (
                                in_bounds and (d_in >= 0) and (d_in < D_eff)
                            )
                        addr = addr_safe if in_bounds else self.num_records

                    comptime if active_warps_this_iter == Self.num_loading_warps:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(addr),
                            smem_ptr,
                            scalar_offset=Int32(0),
                        )
                    else:
                        if Int(self.warp_id) < active_warps_this_iter:
                            self.buffer.load_to_lds[width=Self.load_width](
                                Int32(addr),
                                smem_ptr,
                                scalar_offset=Int32(0),
                            )
            else:
                # Non-swizzle path: m_lane is linear in `i`
                # (m_lane = m_offset + thread_row + i*rows_per_iteration).
                # Hoist (n, h_out, w_out) decomposition out of the loop
                # and increment with carry per iteration. (kh, kw, c_lane)
                # also hoistable: uniform-substrip case computes them
                # once per call from k_offset; per-lane-substrip case
                # computes them once per call from k_offset + thread_col
                # (still per-call constant because thread_col is per-lane
                # constant across iters).
                # step_w / step_h: how (h_out, w_out) advance per iter.
                # Comptime path folds these to constants; runtime path
                # computes them once per `load_tile` call. The
                # `step_h + 1 < H_out` invariant is enforced as a
                # comptime assert on the static path and as a runtime
                # precondition (set by the dispatcher) on the
                # use_runtime_hw path.
                # step_w / step_h / step_d: how (w_out, h_out, d_out)
                # advance per iter when m_lane bumps by
                # rows_per_iteration. step_d is 0 for typical shapes
                # (rows_per_iteration < H_out*W_out). The carry chain
                # below propagates overflow w → h → d → n.
                var step_w: Int
                var step_h: Int
                var step_d: Int = 0
                comptime if Self.use_runtime_hw:
                    step_w = Self.rows_per_iteration % W_out_eff
                    var rest = Self.rows_per_iteration // W_out_eff
                    comptime if Self._is_3d:
                        step_h = rest % H_out_eff
                        step_d = rest // H_out_eff
                    else:
                        step_h = rest
                else:
                    step_w = Self.rows_per_iteration % Self.W_out
                    comptime if Self._is_3d:
                        comptime _rest = (Self.rows_per_iteration // Self.W_out)
                        step_h = _rest % Self.H_out
                        step_d = _rest // Self.H_out
                        # Single-carry invariant: with d_out starting up to
                        # D_out-1 and the h_out→d_out carry contributing +1
                        # in the worst case, the post-add d_out is at most
                        # D_out + step_d. One subtract leaves at most step_d;
                        # we need that in [0, D_out-1], i.e. `step_d <
                        # D_out`. Tight bound — small D_out shapes (e.g.
                        # D_out=2 in test_conv_gpu) require this not be over-
                        # conservative. If a future shape trips even this,
                        # replace the `if` carry with a `while` loop.
                        comptime assert (
                            Self.rows_per_iteration // (Self.W_out * Self.H_out)
                        ) < Self.D_out, (
                            "Incremental m-decomposition (3D) needs"
                            " rows_per_iteration//(W_out*H_out) < D_out"
                        )
                    else:
                        step_h = Self.rows_per_iteration // Self.W_out
                        comptime assert (
                            Self.rows_per_iteration // Self.W_out
                        ) + 1 < Self.H_out, (
                            "Incremental m-decomposition needs"
                            " rows_per_iteration//W_out + 1 < H_out"
                        )

                var m_lane0 = m_eff + self.thread_row
                # Initial M decomposition.
                #   2D: m_lane0 → (n, h_out, w_out)
                #   3D: m_lane0 → (n, d_out, h_out, w_out)
                var n: Int
                var d_out: Int = 0  # 2D: unused
                var h_out: Int
                var w_out: Int
                comptime if Self._is_3d:
                    var n_within_dhw = divmod(m_lane0, spatial_dhw)
                    n = n_within_dhw[0]
                    var m_within_dhw = n_within_dhw[1]
                    var d_within_hw = divmod(m_within_dhw, spatial)
                    d_out = d_within_hw[0]
                    var m_within_hw = d_within_hw[1]
                    var h_w = divmod(m_within_hw, W_out_eff)
                    h_out = h_w[0]
                    w_out = h_w[1]
                else:
                    var n_m_within = divmod(m_lane0, spatial)
                    n = n_m_within[0]
                    var m_within = n_m_within[1]
                    var h_w = divmod(m_within, W_out_eff)
                    h_out = h_w[0]
                    w_out = h_w[1]

                # Pick (kq, kh, kw, c_lane) per the comptime fast/slow choice.
                # kq is only used in 3D mode.
                var kq: Int = 0
                var kh: Int
                var kw: Int
                var c_lane: Int
                comptime if Self._is_uniform_substrip:
                    var substrip_uniform = k_eff // Self.C
                    comptime if Self._is_3d:
                        var rs_uniform = substrip_uniform % (Self.R * Self.S)
                        kq = substrip_uniform // (Self.R * Self.S)
                        kh = rs_uniform // Self.S
                        kw = rs_uniform % Self.S
                    else:
                        kh = substrip_uniform // Self.S
                        kw = substrip_uniform % Self.S
                    c_lane = (k_eff % Self.C) + self.thread_col
                else:
                    var k_lane = k_eff + self.thread_col
                    var substrip_lane = k_lane // Self.C
                    comptime if Self._is_3d:
                        var rs_lane = substrip_lane % (Self.R * Self.S)
                        kq = substrip_lane // (Self.R * Self.S)
                        kh = rs_lane // Self.S
                        kw = rs_lane % Self.S
                    else:
                        kh = substrip_lane // Self.S
                        kw = substrip_lane % Self.S
                    c_lane = k_lane % Self.C

                comptime for i in range(Self.num_iterations):
                    comptime active_warps_this_iter = min(
                        Self.num_loading_warps,
                        Self.total_warp_rows - i * Self.num_loading_warps,
                    )
                    var tile_idx = i * Self.num_loading_warps + self.warp_id
                    var warp_tile = dst.tile[
                        Self.rows_per_warp, Self.tile_cols
                    ](tile_idx, 0)
                    var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                    var h_in = (
                        h_out * Self.stride_h
                        + kh * Self.dilation_h
                        - Self.pad_h
                    )
                    var w_in = (
                        w_out * Self.stride_w
                        + kw * Self.dilation_w
                        - Self.pad_w
                    )
                    var addr_safe: Int
                    var d_in: Int = 0  # 2D: unused
                    comptime if Self._is_3d:
                        d_in = (
                            d_out * Self.stride_d
                            + kq * Self.dilation_d
                            - Self.pad_d
                        )
                        addr_safe = (
                            ((n * D_eff + d_in) * H_eff + h_in) * W_eff + w_in
                        ) * Self.C + c_lane
                    else:
                        addr_safe = (
                            (n * H_eff + h_in) * W_eff + w_in
                        ) * Self.C + c_lane
                    var addr = addr_safe
                    comptime if Self._needs_halo_mask:
                        var in_bounds = (
                            (h_in >= 0)
                            and (h_in < H_eff)
                            and (w_in >= 0)
                            and (w_in < W_eff)
                        )
                        comptime if Self._is_3d:
                            in_bounds = (
                                in_bounds and (d_in >= 0) and (d_in < D_eff)
                            )
                        addr = addr_safe if in_bounds else self.num_records

                    comptime if active_warps_this_iter == Self.num_loading_warps:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(addr),
                            smem_ptr,
                            scalar_offset=Int32(0),
                        )
                    else:
                        if Int(self.warp_id) < active_warps_this_iter:
                            self.buffer.load_to_lds[width=Self.load_width](
                                Int32(addr),
                                smem_ptr,
                                scalar_offset=Int32(0),
                            )

                    # Carry-increment (n, [d_out,] h_out, w_out) for
                    # the next iter. 2D path: w → h → n. 3D path:
                    # w → h → d → n.
                    comptime if i + 1 < Self.num_iterations:
                        w_out += step_w
                        h_out += step_h
                        comptime if Self._is_3d:
                            d_out += step_d
                        if w_out >= W_out_eff:
                            w_out -= W_out_eff
                            h_out += 1
                        if h_out >= H_out_eff:
                            h_out -= H_out_eff
                            comptime if Self._is_3d:
                                d_out += 1
                            else:
                                n += 1
                        comptime if Self._is_3d:
                            if d_out >= D_out_eff:
                                d_out -= D_out_eff
                                n += 1
