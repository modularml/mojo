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
"""Structured Apple M5 simdgroup-tiled matmul (Metal 4 hardware MMA).

Everything lives in the `AppleM5MatMul` struct (mirroring the AMD/NVIDIA
structured kernels): the comptime config, the Morton tile scheduler, the
B-layout helper, the single-pass GPU kernel (`run`), and the split-K kernels
(`run_split_k_partial` / `run_split_k_reduce`). The `enqueue_apple_matmul` /
`enqueue_apple_matmul_split_k` free functions are the host-side launchers
(kept standalone so callers and tests dispatch without naming the struct).

64x64 output tile per threadgroup; four simdgroups (128 threads) each own a
32x32 subtile (2x2 `MmaOpApple`). A per-simdgroup runtime branch picks between
an unbounded fast path and a bounded path for ragged M/N edges and partial K
tails. Operands load DRAM->register directly -- threadgroup-memory staging
*degrades* matmul on Apple Silicon. See `kernels/apple-m5-matmul` in the KB.
"""

from std.collections import Array, Optional
from max.gpu import WARP_SIZE, block_dim, block_idx, lane_id, thread_idx
from std.math import align_down, ceildiv
from max.gpu.host import DeviceContext
from std.sys import align_of, size_of
from std.utils import IndexList
from layout import TensorEngine, TileTensor, Idx
from layout.tile_layout import Layout, TensorLayout, row_major
from layout.coord import Coord
from linalg.arch.apple.mma import ConvIm2colParams, MmaOpApple
from linalg.utils import elementwise_epilogue_type


# === A-operand loader abstraction ========================================== #
# `run` (plain GEMM) and `run_conv` (fused online-im2col conv) share the WHOLE
# simdgroup-tiled GEMM body (`_run_gemm_body`); the ONLY divergence is how each
# BK strip's A side is produced: `run` reads a contiguous `[M, K]` slab,
# `run_conv` gathers the im2col fragment from NHWC on the fly. That single seam
# lives behind a comptime loader trait + two impls, so the body is written once
# and specialized at zero cost (loader is a comptime param, methods inline).
#
# This mirrors the AMD MI355 `TileLoaderLDS` vs `TileLoaderLDSIm2col` split (KB
# `new-primitives/amd-tile-io-expert-objects`: "every transition has an owner";
# zero abstraction cost), except the Apple seam is the MMA FRAGMENT LOAD, not an
# LDS stage -- Apple matmul has no shared-memory staging (KB
# `patterns/apple-m5-gpu-performance-considerations` -- staging *degrades* it).


# The body's MMA op: `accum_type` accumulation, the kernel's `in_type` A operand,
# a possibly-distinct `b_type` B/weight operand, and the simdgroup MMA-fragment
# count `num_m x num_n` (= `SG_M/16 x SG_N/16`). Default 2x2 is the dense-GEMM
# optimum; the conv path uses 1x1 (16x16 simdgroup tile). `b_type` and
# `accum_type` default to the dense/conv case (B == A dtype, fp32 accumulate), so
# existing instantiations are unchanged; the W8A16 FP8 path sets `b_type=fp8`
# (the B/weight-loader seam swap) and a later int8 slice sets `accum_type=int32`.
# Type-identical to `AppleM5MatMul.Mma` for the same counts (the two MUST stay in
# sync). `AccumType` is `Array[SIMD[accum_type, 8], num_m*num_n]`, fixed by
# the accum out-type and the fragment count, NOT by `in_type`.
comptime _BodyMma[
    in_type: DType,
    num_m: Int = 2,
    num_n: Int = 2,
    b_type: DType = in_type,
    accum_type: DType = .float32,
] = MmaOpApple[
    accum_type, in_type, num_m_mmas=num_m, num_n_mmas=num_n, b_type=b_type
]


trait AOperandLoader:
    """One BK-strip A contribution for the shared Apple GEMM body.

    `accumulate_strip` is the single seam between the plain-GEMM and fused-conv
    kernels: for strip index `k_strip`, it loads this strip's A side (slab tile
    vs online im2col gather) and runs the MMA into `accum`. Both impls consume
    the same pre-tiled `b_sub` and write the same `accum`; only the A side
    differs. Any per-simdgroup slab is built once by the caller before the
    K-loop and held in the impl (see `DenseALoader`), so `accumulate_strip` only
    adds this strip's K offset (`k_strip * BK`).

    The loader is parametrized on the GEMM `in_type`, so the seam method names
    the concrete `_BodyMma[Self.in_type]` / its `AccumType`. The shared body
    infers the loader's CONCRETE type (infer-only `L`) and keys its own
    `mma_op`/`accum`/B on `L.in_type` so both sides of the erased dispatch agree
    (KB `patterns/trait-type-erasure-and-stride-layout-workaround`); at the
    `run`/`run_conv` call sites `L.in_type == Self.in_type` is proven, so there
    is no runtime cost.
    """

    comptime in_type: DType
    # B/weight operand dtype fed to the MMA. Equals `in_type` for dense/conv;
    # the W8A16 FP8 path sets it to `float8_e4m3fn` (the native fp8 MMA B operand,
    # the "weight-loader seam" swap). The A loader still ISSUES the MMA (conv
    # fuses the A-gather into `mma_im2col`, so MMA ownership stays A-side); this
    # associated dtype only widens the `b_sub`/`mma_op` types it forwards.
    comptime b_type: DType
    # Simdgroup MMA-fragment counts (= SG_M/16, SG_N/16); the body keys its
    # `mma_op`/`accum` on these so loader and body agree on the accumulator size.
    comptime num_m_mmas: Int
    comptime num_n_mmas: Int

    @always_inline
    def accumulate_strip[
        bounded: Bool, b_layout: TensorLayout
    ](
        mut self,
        mut mma_op: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ],
        mut accum: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ].AccumType,
        b_sub: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin],
        conv: ConvIm2colParams,
        k_strip: Int32,
        *,
        a_valid_rows: Int32,
        b_valid_cols: Int,
        k_valid: Int,
    ):
        # `mut self`: the conv loader carries the strength-reduced K-state
        # (c0/r/s) and advances it per strip; the dense loader is stateless.
        ...


@fieldwise_init
struct DenseALoader[
    dtype: DType,
    a_layout: TensorLayout,
    BK: Int = 16,
    SG_M: Int = 32,
    use_x2: Bool = False,
    b_dtype: DType = dtype,
](AOperandLoader, ImplicitlyCopyable, Movable):
    """Plain-GEMM A loader: holds the pre-tiled `(SG_M, K)` slab.

    The `run` wrapper bakes the simdgroup's slab base once (see its docstring for
    the hoist rationale); the hot-loop `.tile[SG_M, BK](0, k_strip)` here only
    adds this strip's `k_strip * BK` K offset. The strip sub-tiles are
    type-identical to the per-strip form, so `MmaOpApple.mma` is unchanged.

    `BK`/`SG_M` are explicit params (not re-derived from a fresh default
    `AppleM5MatMul[dtype]`) so a non-default `block_k`/`sg_m` -- e.g. `use_x2`'s
    `block_k=32` -- isn't silently ignored; `run` passes `Self.BK`/`Self.SG_M`.

    `b_dtype` sets the B/weight operand fed to the MMA (defaults to `dtype`, the
    dense case). The W8A16 FP8 path constructs this loader with `b_dtype=fp8`:
    the A slab stays bf16 and only `b_sub` / `mma_op` widen to the native-fp8 MMA
    (KGEN lowers the fp8 B fragment to AIR `<8 x i8>`); the K-loop and
    accumulation order are byte-for-byte the dense path.

    Lifetime: the `a_slab` view is held with `UntrackedOrigin` (struct fields
    cannot expose an `AnyOrigin`); the kernel arg it derives from outlives the
    K-loop, so this is the explicit-lifetime case the field-origin rule allows.

    Parameters:
        dtype: Element type of the A operand (fp16, bf16, fp32).
        a_layout: `TensorLayout` of the pre-tiled A slab held by the loader.
        BK: K-strip depth per accumulate step; must match the body's `BK`
            tiling (defaults to 16).
        SG_M: Simdgroup subtile rows `SG_M`; the `.tile[SG_M, BK]` extent
            used per strip (defaults to 32).
        use_x2: Use the NT "double-strip" dense MMA (`mma_dense_x2`,
            `block_k=32`) instead of the single-strip `mma` (defaults to
            False).
        b_dtype: Element type of the B/weight operand fed to the MMA
            (defaults to `dtype`, the dense case).
    """

    # Satisfy the trait's associated `in_type` from the struct's `dtype` param.
    comptime in_type = Self.dtype
    # B operand dtype (trait `b_type`), from the struct's `b_dtype` param.
    comptime b_type = Self.b_dtype
    # Dense GEMM is fixed at the 2x2 (SG=32) simdgroup optimum.
    comptime num_m_mmas = 2
    comptime num_n_mmas = 2

    var a_slab: TileTensor[Self.dtype, Self.a_layout, ImmUntrackedOrigin]

    @always_inline
    def accumulate_strip[
        bounded: Bool, b_layout: TensorLayout
    ](
        mut self,  # stateless for dense; `mut` only to satisfy the trait
        mut mma_op: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ],
        mut accum: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ].AccumType,
        b_sub: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin],
        conv: ConvIm2colParams,  # unused by the dense path
        k_strip: Int32,
        *,
        a_valid_rows: Int32,
        b_valid_cols: Int,
        k_valid: Int,
    ):
        var a_sub = self.a_slab.tile[Self.SG_M, Self.BK](0, Int(k_strip))
        comptime if Self.use_x2:
            mma_op.mma_dense_x2[bounded=bounded](
                accum,
                a_sub,
                b_sub,
                a_valid_rows=Int(a_valid_rows),
                b_valid_cols=b_valid_cols,
                k_valid=k_valid,
            )
        else:
            mma_op.mma[bounded=bounded](
                accum,
                a_sub,
                b_sub,
                a_valid_rows=Int(a_valid_rows),
                b_valid_cols=b_valid_cols,
                k_valid=k_valid,
            )


struct Im2colALoader[
    dtype: DType,
    BK: Int = 16,
    num_m: Int = 2,
    num_n: Int = 2,
    c_aligned: Bool = False,
](AOperandLoader, ImplicitlyCopyable, Movable):
    """Fused-conv A loader: `input_ptr` + this simdgroup's prebaked pixel
    anchors and carried K-state. The gather reads the A MMA-fragment from NHWC
    via `MmaOpApple.mma_im2col` (the im2col matrix is non-affine, so it is not a
    `distribute`-expressible TileTensor -- KB
    `exceptions/apple-mma-fragment-is-not-distribute-expressible`). The anchor
    prebake + K-state strength-reduction design: KB `kernels/apple-conv2d-im2col`.

    `conv` is deliberately NOT a struct field: an 11-`Int32` `ConvIm2colParams`
    held in a by-value loader spilled to a GENERIC addrspace alloca that Metal's
    AIR backend mishandles (the FLUX-concat addrspace-loss class) and crashed
    MTLCompilerService -- so it is threaded as an `accumulate_strip` arg. The
    prebaked anchors are plain `Int32` arrays (safe, like `DenseALoader`'s slab).

    Lifetime: `input_ptr` uses `UntrackedOrigin` (struct fields cannot expose
    `AnyOrigin`); the NHWC input kernel arg outlives the K-loop gather.

    Parameters:
        dtype: Element type of the NHWC input operand (bf16 for now).
        BK: K-strip depth per accumulate step; must match the body's `BK`
            tiling (defaults to 16).
        num_m: M MMA-fragment count per simdgroup, `SG_M / 16` (defaults to 2).
        num_n: N MMA-fragment count per simdgroup, `SG_N / 16` (defaults to 2).
        c_aligned: When True, assume `conv.C` is a multiple of 8 so the
            channel run is contiguous and a single width-8 load suffices
            (defaults to False).
    """

    comptime in_type = Self.dtype
    # Conv B (the filter) is the same dtype as A; no fp8/int8 conv path yet.
    comptime b_type = Self.dtype
    comptime num_m_mmas = Self.num_m
    comptime num_n_mmas = Self.num_n
    comptime NUM_ROWS = Self.num_m * 2  # 2 row-halves x num_m M-fragments

    var input_ptr: UnsafePointer[Scalar[Self.dtype], ImmUntrackedOrigin]
    var h_base: Array[Int32, Self.NUM_ROWS]
    var w_base: Array[Int32, Self.NUM_ROWS]
    var batch_base: Array[Int32, Self.NUM_ROWS]
    # Carried K-state (k0base -> r, s, c0): seeded in `__init__`, advanced per
    # strip by add+carry instead of `//`/`%` (KB `kernels/apple-conv2d-im2col`).
    var c0: Int32
    var r: Int32
    var s: Int32
    var k_total: Int32  # R*S*C, prebaked once (the partial-K bound)

    def __init__(out self, *, copy: Self):
        self.input_ptr = copy.input_ptr
        self.h_base = copy.h_base.copy()
        self.w_base = copy.w_base.copy()
        self.batch_base = copy.batch_base.copy()

        self.c0 = copy.c0
        self.r = copy.r
        self.s = copy.s
        self.k_total = copy.k_total

    @always_inline
    def __init__(
        out self,
        input_ptr: UnsafePointer[Scalar[Self.dtype], ImmUntrackedOrigin],
        row_base: Int32,
        rb: Int32,
        cb: Int32,
        conv: ConvIm2colParams,
    ):
        """Prebake this lane's per-row im2col anchors + seed the K-state.

        `row_base` is the simdgroup's absolute M base (`_sg_row_base`); `rb`/`cb`
        are this lane's MMA-fragment row/col (matching `MmaOpApple.rb`/`.cb`).
        Row `ri` covers M-fragment `ri // 2`, row-half `ri % 2`, i.e. output
        pixel `row_base + (ri//2)*16 + (ri%2)*8 + rb`. The K-state is decomposed
        once here for the lane's first strip (`k0base = 2*cb`); every later strip
        advances it incrementally in `accumulate_strip`.

        Args:
            input_ptr: Pointer to the 4-D NHWC input tensor; the im2col gather
                reads A MMA-fragments from it.
            row_base: Simdgroup's absolute M-row base (`_sg_row_base`); anchors
                this lane's output pixels.
            rb: This lane's MMA-fragment row offset within the simdgroup
                (matches `MmaOpApple.rb`).
            cb: This lane's MMA-fragment column offset within the simdgroup
                (matches `MmaOpApple.cb`); seeds the K-state at `k0base = 2*cb`.
            conv: Conv geometry for the im2col gather; threaded as a value arg,
                not a struct field (see the struct docstring for the Metal
                addrspace reason).
        """
        self.input_ptr = input_ptr
        self.h_base = Array[Int32, Self.NUM_ROWS](uninitialized=True)
        self.w_base = Array[Int32, Self.NUM_ROWS](uninitialized=True)
        self.batch_base = Array[Int32, Self.NUM_ROWS](uninitialized=True)
        var HW_out = conv.H_out * conv.W_out
        var hwc = conv.H * conv.W * conv.C
        comptime for ri in range(Self.NUM_ROWS):
            comptime mi = ri // 2
            comptime half = ri % 2
            var m = row_base + Int32(mi * 16 + half * 8) + rb
            var batch = m // HW_out
            var spatial = m % HW_out
            var h_out = spatial // conv.W_out
            var w_out = spatial % conv.W_out
            self.h_base[ri] = h_out * conv.stride_h - conv.pad_h
            self.w_base[ri] = w_out * conv.stride_w - conv.pad_w
            self.batch_base[ri] = batch * hwc

        # Seed the K-state for k_strip=0: k0base = 2*cb. These are the ONLY
        # divides on the K axis -- later strips advance by add+carry.
        var SC = conv.S * conv.C
        self.k_total = conv.R * SC  # R*S*C, reused as the partial-K bound
        var k0 = Int32(2) * cb
        self.r = k0 // SC
        var sc = k0 % SC
        self.s = sc // conv.C
        self.c0 = sc % conv.C

    @always_inline
    def accumulate_strip[
        bounded: Bool, b_layout: TensorLayout
    ](
        mut self,
        mut mma_op: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ],
        mut accum: _BodyMma[
            Self.in_type,
            Self.num_m_mmas,
            Self.num_n_mmas,
            b_type=Self.b_type,
        ].AccumType,
        b_sub: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin],
        conv: ConvIm2colParams,
        k_strip: Int32,
        *,
        a_valid_rows: Int32,
        b_valid_cols: Int,
        k_valid: Int,
    ):
        # The gather's absolute K origin is invariantly `k_strip * BK`. `BK` is
        # this loader's own param (matches the body's `Self.BK` tiling), NOT a
        # default `AppleM5MatMul[dtype]` instantiation -- the conv path overrides
        # BK, so reading the default here would desync the gather from the tiles.
        var k_base = k_strip * Int32(Self.BK)
        # `bounded` from the body (once per simdgroup) lets interior full strips
        # skip the ragged-M / partial-K branches.
        mma_op.mma_im2col[bounded=bounded, c_aligned=Self.c_aligned](
            accum,
            self.input_ptr,
            conv,
            b_sub,
            self.h_base,
            self.w_base,
            self.batch_base,
            self.c0,
            self.r,
            self.s,
            k_base=k_base,
            m_valid=a_valid_rows,
            k_total=self.k_total,
            b_valid_cols=b_valid_cols,
            k_valid=k_valid,
        )

        # Advance the K-state to the next strip: k0base += BK, renormalized into
        # (r, s, c0) by add+carry (no divides). For C >= BK the c0/s carries run
        # at most once each; smaller C loops a bounded number of times.
        self.c0 += Int32(Self.BK)
        while self.c0 >= conv.C:
            self.c0 -= conv.C
            self.s += 1
        while self.s >= conv.S:
            self.s -= conv.S
            self.r += 1


# === B-operand (weight) loader policy ====================================== #
# The B-side dual of `AOperandLoader`. Where `AOperandLoader` owns the A side of
# each BK strip (and ISSUES the MMA -- conv fuses its A-gather into `mma_im2col`,
# so MMA ownership must stay A-side to host conv on the shared body), this policy
# owns the B/weight side: the operand dtype, the accumulator dtype, and whether a
# cooperative threadgroup-memory (SMEM) staging pass is needed.
#
# For the 16x16x16 `MmaOpApple` formats whose B is a native MMA operand
# (dense-bf16, FP8-W8A16, int8-W8A8) the B strip is a direct DRAM sub-tile the
# body builds inline and the A loader feeds to the MMA -- so those set
# `needs_smem=False` and the shared body allocates ZERO threadgroup memory for
# them (verified via the FP8 AIR dump). Only the FP4 W4A16 cooperative-dequant
# variant needs SMEM staging (`needs_smem=True`, a later slice); the body will
# guard its `stack_allocation` + barrier + post-decode OOB handling behind
# `comptime if W.needs_smem` so the fp8/int8/dense fast paths never pay for it.
# Mirrors the AMD tile-io write/decode owners (KB
# `new-primitives/amd-tile-io-expert-objects`: "every DRAM<->register transition
# has an owner").
trait WeightLoader:
    """B/weight-side comptime policy for the shared Apple GEMM body.

    Carries the B operand dtype (`b_type`), the MMA accumulator dtype
    (`accum_type`; fp32 for dense/fp8/fp4, int32 for a later int8 slice), and the
    SMEM-staging policy (`needs_smem` / `smem_elems`). The A loader still issues
    the MMA; this policy tells the body how to source and size the B side.
    """

    # B/weight operand dtype (native MMA operand: bf16 / fp8 / int8 / -- FP4 uses
    # packed uint8 + a cooperative dequant, a later slice).
    comptime b_type: DType
    # MMA accumulator dtype (fp32 dense/fp8/fp4; int32 for int8 in a later slice).
    comptime accum_type: DType
    # True only for cooperative-SMEM formats (FP4); False for dense/fp8/int8, for
    # which the body allocates NO threadgroup memory.
    comptime needs_smem: Bool
    # SMEM staging element count (0 when `needs_smem` is False).
    comptime smem_elems: Int


@fieldwise_init
struct DenseWeightLoader[b_dtype: DType, accum_dtype: DType = .float32](
    ImplicitlyCopyable, Movable, WeightLoader
):
    """Direct-DRAM B policy: dense-bf16 and FP8-W8A16.

    The B strip is a plain `(BK, SG_N)` DRAM sub-tile (coalesced via the
    col-major `(K, N)` view for `transpose_b=True`) fed straight to the native
    MMA -- no staging. `b_dtype == in_type` is the dense case; `b_dtype=fp8` is
    the W8A16 weight-only case (the only divergence from dense is the operand
    dtype). Stateless: pure comptime policy.
    """

    comptime b_type = Self.b_dtype
    comptime accum_type = Self.accum_dtype
    comptime needs_smem = False
    comptime smem_elems = 0


struct AppleM5MatMul[
    in_type: DType,
    c_type: DType = .float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
    k_unroll: Int = 1,
    use_x2: Bool = False,
    linear_idx_type: DType = .int64,
    clamp_edge: Bool = False,
    b_type: DType = in_type,
    accum_type: DType = .float32,
]:
    """Apple M5 simdgroup-tiled GEMM (Metal 4 hardware MMA).

    Parameters:
        in_type: A element type (fp16, bf16, fp32; int8 at the MMA-op level).
        c_type: Output element type (fp16, bf16, fp32). Accumulation is
            `accum_type`.
        transpose_b: If True, B is `(N, K)` row-major (viewed as `col_major(K, N)`);
            otherwise B is `(K, N)` row-major.
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract).
        block_m: Threadgroup block rows `BM` (M tile). Multiple of `SG_M`.
        block_n: Threadgroup block cols `BN` (N tile). Multiple of `SG_N`.
        block_k: K-strip depth `BK` per accumulate step. Multiple of `MMA_K`
            (16). The conv path (`enqueue_apple_conv2d`) overrides these to tune
            the fused im2col GEMM independently of the dense GEMM defaults.
        sg_m: Simdgroup subtile rows `SG_M`. Multiple of `MMA_M` (16); fixes the
            M MMA-fragment count `NUM_MMA_M = SG_M / 16`.
        sg_n: Simdgroup subtile cols `SG_N`. Multiple of `MMA_N` (16); fixes the
            N MMA-fragment count `NUM_MMA_N = SG_N / 16`.
        k_unroll: Dense `run` path interior K-loop unroll factor (default 1, no
            unroll). Processes `k_unroll` `BK`-strips per loop pass, amortizing
            loop-control overhead across the group. Regresses severely if
            combined with `use_x2` (register pressure) -- the dispatcher never
            sets both.
        use_x2: NT bf16/fp16 "double-strip" dense MMA (`block_k=32`,
            `MmaOpApple.mma_dense_x2`): both operands are K-contiguous, so one
            call covers 32 K, halving the K-loop iteration count.
        linear_idx_type: Integer type for the dense `run` path's A/B/C
            `TileTensor` offset arithmetic (Apple's scalar ALU is faster on
            32-bit). Only safe when every offset provably fits int32.
        clamp_edge: Enables `clamp_v2`: shift a ragged last M/N tile to
            `m-BM`/`n-BN` for a full, fast load, then store back only its
            owned region so it doesn't overwrite the previous tile. Dense
            fp32 only, no `elementwise_lambda_fn`. Only `run_chained`
            consumes it; `run`/`run_conv` never set it.
        b_type: B/weight operand dtype fed to the MMA. Defaults to `in_type` (the
            dense/conv case). The W8A16 FP8 path sets `b_type=float8_e4m3fn`: the
            A slab stays `in_type` (bf16) and only the B fragment widens to the
            native-fp8 MMA operand (the weight-loader seam swap). See
            `WeightLoader` / `enqueue_matmul2d_fp8`.
        accum_type: MMA accumulator dtype. fp32 for dense/fp8; a later int8 slice
            uses int32.

    `run` is the GPU kernel entry (TileTensor operands; `M`/`N`/`K` derived from
    them). `run_split_k_partial` / `run_split_k_reduce` are the split-K kernels
    (bf16 family; split-K is not wired for `b_type != in_type`). Launch via
    `enqueue_apple_matmul` / `enqueue_apple_matmul_split_k`; the FP8 W8A16 dense
    launcher is `enqueue_matmul2d_fp8`.
    """

    # === Tile config. BM/BN/BK come from the block_* params (default 64x64
    # block, BK=16 K-strip); SG_M/SG_N from the sg_* params (default 32 = 2*MMA).
    # Kept as comptime members so the body can alias them. ===
    comptime BM = Self.block_m
    comptime BN = Self.block_n
    comptime BK = Self.block_k
    comptime SG_M = Self.sg_m  # simdgroup subtile rows (NUM_MMA_M * MMA_M)
    comptime SG_N = Self.sg_n  # simdgroup subtile cols (NUM_MMA_N * MMA_N)
    comptime NUM_MMA_M = Self.SG_M // 16  # M MMA fragments per simdgroup
    comptime NUM_MMA_N = Self.SG_N // 16  # N MMA fragments per simdgroup
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE
    comptime REDUCE_BLOCK = 256  # threads/block for the split-K reduce kernel
    # 2x2 (4 accumulators/simdgroup) is the benchmarked dense-GEMM optimum on M5
    # Max; 2x4 / 4x2 regress 26-60% -- the MMA pipeline is not latency-starved at
    # 4 accumulators, so more only adds register pressure / spills. The conv path
    # overrides sg_m/sg_n (and thus this count) via `enqueue_apple_conv2d`.
    comptime Mma = MmaOpApple[
        Self.accum_type,
        Self.in_type,
        num_m_mmas=Self.NUM_MMA_M,
        num_n_mmas=Self.NUM_MMA_N,
        b_type=Self.b_type,
    ]

    # === Morton (Z-order) tile scheduling ================================== #

    @staticmethod
    def morton_decode_2d(flat_idx: UInt32) -> Tuple[UInt32, UInt32]:
        """Decode a linear index to (tile_m, tile_n) via Morton Z-order.

        Even bits of flat_idx -> tile_n, odd bits -> tile_m. The decoded pair
        may fall outside any rectangular grid that isn't a power-of-2 square;
        the caller checks bounds.

        Future: as of AIR 4.1 a `bit_interleave` intrinsic can do this
        interleave directly, replacing the hand-rolled bit-scatter below. A
        Gilbert-curve dispatch order was also explored for better locality,
        but it needs a per-shape predispatch kernel to generate the order.

        Args:
            flat_idx: The linear threadgroup index to split into interleaved
                Z-order bits; even bits form `tile_n`, odd bits form `tile_m`.
        """
        var x = flat_idx & 0x55555555
        var y = (flat_idx >> 1) & 0x55555555

        x = (x | (x >> 1)) & 0x33333333
        x = (x | (x >> 2)) & 0x0F0F0F0F
        x = (x | (x >> 4)) & 0x00FF00FF
        x = (x | (x >> 8)) & 0x0000FFFF

        y = (y | (y >> 1)) & 0x33333333
        y = (y | (y >> 2)) & 0x0F0F0F0F
        y = (y | (y >> 4)) & 0x00FF00FF
        y = (y | (y >> 8)) & 0x0000FFFF

        return (y, x)

    @staticmethod
    def morton_decode_2d_rect(
        flat_idx: UInt32,
        log2_m: UInt32,
        log2_n: UInt32,
    ) -> Tuple[UInt32, UInt32]:
        """Decode `flat_idx` to (tile_m, tile_n) over a `(1<<log2_m) x (1<<log2_n)` grid.

        Z-order covers a `min(side_m, side_n)` square core; remaining bits sweep
        the longer axis. Reduces to `morton_decode_2d` when `log2_m == log2_n`.

        Args:
            flat_idx: The linear threadgroup index to decode into a 2-D tile
                coordinate.
            log2_m: Base-2 logarithm of the grid's M extent; the decoded
                `tile_m` ranges over `[0, 1<<log2_m)`.
            log2_n: Base-2 logarithm of the grid's N extent; the decoded
                `tile_n` ranges over `[0, 1<<log2_n)`.
        """
        var log2_lo = min(log2_m, log2_n)
        var lo_mask = (UInt32(1) << (UInt32(2) * log2_lo)) - UInt32(1)
        var lo_mn = Self.morton_decode_2d(flat_idx & lo_mask)
        var hi_bits = flat_idx >> (UInt32(2) * log2_lo)

        var m_extra: UInt32 = (
            hi_bits << log2_lo
        ) if log2_m > log2_n else UInt32(0)
        var n_extra: UInt32 = (
            hi_bits << log2_lo
        ) if log2_n > log2_m else UInt32(0)

        return (lo_mn[0] | m_extra, lo_mn[1] | n_extra)

    # === B-operand layout selection ======================================= #

    comptime _PickBMatLayoutResult: TensorLayout = type_of(
        Layout(Coord(Int(), Int()), Coord(Idx[1], Int()))
    ) if Self.transpose_b else type_of(
        Layout(Coord(Int(), Int()), Coord(Int(), Idx[1]))
    )

    @staticmethod
    def _pick_b_mat_layout(
        k: Int,
        n: Int,
        out result: Self._PickBMatLayoutResult,
    ):
        """Full B=(K, N) Layout selected at comptime.

        `transpose_b=True`  -> strides `(1, K)` (col_major view of an (N,K) buffer).
        `transpose_b=False` -> strides `(N, 1)` (row_major (K, N)).

        `run` pre-tiles this into a per-simdgroup SG_N-wide column slab
        (full K), then the K-loop tiles only the K axis; no pointer arithmetic.
        """
        comptime if Self.transpose_b:
            return rebind[type_of(result)](
                Layout(Coord(k, n), Coord(Idx[1], k))
            )
        else:
            return rebind[type_of(result)](
                Layout(Coord(k, n), Coord(n, Idx[1]))
            )

    # === Shared simdgroup-tiled GEMM body ================================== #
    # `run` and `run_conv` share this whole body; the comptime
    # `loader.accumulate_strip(...)` seam in the K-loop is the only divergence
    # (see the "A-operand loader abstraction" header above).

    @always_inline
    @staticmethod
    def _sg_row_base(log2_grid_m: UInt32, log2_grid_n: UInt32) -> Int32:
        """This simdgroup's absolute M-row base: `tile_m*BM + sg_m_idx*SG_M`.

        Used by the `run` wrapper to build the loop-invariant A slab BEFORE the
        body runs. `_run_gemm_body` computes the SAME quantity inline from its
        own already-decoded `tile_m`/`sg_m_idx` (calling this helper there would
        redundantly re-run the Morton decode -- measurably slower on small conv
        shapes). The two must agree: `BM` for the tile term, `SG_M` for the
        simdgroup term -- they differ (BM=64, SG_M=32); mixing them up is a
        per-tile slab-base bug invisible on single-tile shapes.
        """
        var tile_mn = Self.morton_decode_2d_rect(
            UInt32(block_idx.x), log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // Int32(Self.NUM_SG_N)
        return tile_m * Int32(Self.BM) + sg_m_idx * Int32(Self.SG_M)

    @always_inline
    @staticmethod
    def _run_gemm_body[
        L: AOperandLoader,
        //,
        W: WeightLoader,
        c_layout: TensorLayout,
        b_layout: TensorLayout,
        seed_from_output: Bool = False,
    ](
        mut loader: L,
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=_],
        b: TileTensor[L.b_type, b_layout, ImmutAnyOrigin, Engine=_],
        k: Int,
        conv: ConvIm2colParams,
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
        k_strip_start: Int = 0,
        k_strip_end: Int = -1,
    ):
        """Shared GEMM body; A side from `loader`, B policy from `W`.

        C is `(M, N)` row-major (`M`/`N` derive from `c`); `k` is the contraction
        extent (`a.dim[1]` for plain GEMM, `R*S*C` for conv). B is `(K, N)` for
        `transpose_b=False` or `(N, K)` for `transpose_b=True`, in `Self.b_type`.
        Grid is `(1<<log2_grid_m) * (1<<log2_grid_n)` threadgroups of 128 threads;
        OOB threadgroups early-return after Morton decode.

        `W: WeightLoader` is the B-side policy (operand/accumulator dtype + SMEM
        staging). `loader` produces the A side and ISSUES the MMA; for the direct
        -DRAM B formats (dense/fp8) the body builds the `(BK, SG_N)` B sub-tile
        inline and feeds it to `loader.accumulate_strip`, byte-for-byte the dense
        path with only `Self.b_type` widened. `W.needs_smem` gates the (future)
        cooperative-SMEM decode; fp8/dense set it False so NO threadgroup memory
        is allocated.

        `seed_from_output`/`k_strip_start`/`k_strip_end` support
        `run_chained`'s 2-pass dispatch; other callers (`run`, `run_conv`)
        use the defaults for a single full-K pass.
        """
        comptime assert (
            W.b_type == Self.b_type and W.accum_type == Self.accum_type
        ), "WeightLoader policy must match the kernel's b_type / accum_type"
        # The WeightLoader carries `accum_type` as the B-side seam contract, but
        # this shared body accumulates in fp32 (dense/conv/fp8) -- int32
        # accumulation (int8 W8A8) generalizes the A-loader + epilogue in a later
        # slice, so keep the body fp32-only until then.
        comptime assert (
            Self.accum_type == .float32
        ), "shared GEMM body is fp32-accumulate; int32 accum is a later slice"
        comptime assert (
            L.b_type == Self.b_type
        ), "A-loader b_type must match the kernel's b_type"
        var c_ptr = c.ptr
        var b_ptr = b.ptr
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())

        # Apple's scalar ALU is faster on 32-bit math; use Int32 locally and
        # cast back to Int only at API boundaries (.tile[], MmaOpApple).
        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N
        # Key the MMA op entirely on the LOADER `L` (A `in_type` AND B `b_type`):
        # the erased trait dispatch only sees `L`'s comptimes, so both the body
        # and `accumulate_strip` must agree on the SAME symbols -- keying B on
        # `L.b_type` (not `W.b_type`/`Self.b_type`, which are distinct symbols the
        # checker won't unify across the trait boundary) is the type-erasure
        # discipline (KB `patterns/trait-type-erasure-and-stride-layout-workaround`).
        # `L.b_type == W.b_type == Self.b_type` by construction (asserted above).
        # Accumulation is fp32 (the body's `accum_type` default). Type-identical
        # to `Self.Mma`.
        comptime Mma = _BodyMma[
            L.in_type,
            L.num_m_mmas,
            L.num_n_mmas,
            b_type=L.b_type,
        ]
        # `clamp_edge` only applies on the fast fp32 store path, never with
        # the cast/lambda epilogue below (see struct docstring).
        comptime clamp_active = Self.clamp_edge and not (
            Self.c_type != .float32 or Self.elementwise_lambda_fn
        )
        # `c_type` / `elementwise_lambda_fn` / `transpose_b` are struct *params*:
        # spelled `Self.x` below (a param can't be aliased to a same-name local
        # the way the members just above are).

        var m_i32 = Int32(m)
        var n_i32 = Int32(n)
        var k_i32 = Int32(k)
        comptime BM_i32: Int32 = Int32(BM)
        comptime BN_i32: Int32 = Int32(BN)
        comptime BK_i32: Int32 = Int32(BK)
        comptime SG_M_i32: Int32 = Int32(SG_M)
        comptime SG_N_i32: Int32 = Int32(SG_N)
        comptime NUM_SG_N_i32: Int32 = Int32(Self.NUM_SG_N)

        var grid_m = ceildiv(m_i32, BM_i32)
        var grid_n = ceildiv(n_i32, BN_i32)

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // NUM_SG_N_i32
        var sg_n_idx = sg_id % NUM_SG_N_i32

        var flat_idx = UInt32(block_idx.x)

        var tile_mn = Self.morton_decode_2d_rect(
            flat_idx, log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var mma_op = Mma()

        # `row_base` inline from the already-decoded `tile_m`/`sg_m_idx` -- MUST
        # match `_sg_row_base` (which the `run` wrapper uses to pre-tile the
        # slab); inlined here to avoid its redundant Morton re-decode (see that
        # helper's docstring).
        var row_base = tile_m * BM_i32 + sg_m_idx * SG_M_i32
        var col_base = tile_n * BN_i32 + sg_n_idx * SG_N_i32
        var sg_row_idx = row_base // SG_M_i32
        var sg_col_idx = col_base // SG_N_i32

        var sg_m_end = row_base + SG_M_i32
        var sg_n_end = col_base + SG_N_i32
        var is_edge_tile = (sg_m_end > m_i32) or (sg_n_end > n_i32)

        # A clamped tile's later simdgroups can have row_base/col_base >= m/n
        # before the shift is applied, so the early return below must not
        # reject them on the clamped axis.
        var is_clamp_row = False
        var is_clamp_col = False
        comptime if clamp_active:
            is_clamp_row = (tile_m * BM_i32 + BM_i32) > m_i32
            is_clamp_col = (tile_n * BN_i32 + BN_i32) > n_i32

        # Skip fully-OOB simdgroups: there's no later threadgroup-uniform op,
        # so the early return is safe, except on a clamped axis where the
        # shift can still bring a simdgroup back in bounds.
        if (row_base >= m_i32 and not is_clamp_row) or (
            col_base >= n_i32 and not is_clamp_col
        ):
            return

        # clamp_v2 shift and single-writer store ownership (see `clamp_edge`
        # docstring). No-op below when `clamp_active` is False.
        var row_shift = Int32(0)
        var col_shift = Int32(0)
        var v2_valid_rows = SG_M_i32
        var v2_valid_cols = SG_N_i32
        comptime if clamp_active:
            var tile_row0 = tile_m * BM_i32
            var tile_col0 = tile_n * BN_i32
            # An aligned last tile has no clamped neighbor to yield to.
            # Skipping this guard corrupts aligned shapes' last tile.
            var row_ragged = (m_i32 % BM_i32) != 0
            var col_ragged = (n_i32 % BN_i32) != 0
            var is_row_neighbor = (
                row_ragged
                and not is_clamp_row
                and (tile_row0 + BM_i32 > m_i32 - BM_i32)
            )
            var is_col_neighbor = (
                col_ragged
                and not is_clamp_col
                and (tile_col0 + BN_i32 > n_i32 - BN_i32)
            )
            if is_clamp_row:
                row_shift = (m_i32 - BM_i32) - tile_row0
            if is_clamp_col:
                col_shift = (n_i32 - BN_i32) - tile_col0

            var row_limit = m_i32 if is_clamp_row else (
                (m_i32 - BM_i32) if is_row_neighbor else (tile_row0 + BM_i32)
            )
            var col_limit = n_i32 if is_clamp_col else (
                (n_i32 - BN_i32) if is_col_neighbor else (tile_col0 + BN_i32)
            )
            var eff_row_base = row_base + row_shift
            var eff_col_base = col_base + col_shift
            v2_valid_rows = max(
                Int32(0), min(SG_M_i32, row_limit - eff_row_base)
            )
            v2_valid_cols = max(
                Int32(0), min(SG_N_i32, col_limit - eff_col_base)
            )

            # A clamped tile is fully in-bounds after the shift, so it takes
            # the fast unbounded K-loop below; only the store stays bounded.
            if is_clamp_row or is_clamp_col:
                is_edge_tile = False

        var k_full_strips = k_i32 // BK_i32
        var has_k_tail = (k_i32 % BK_i32) != 0
        # `k_strip_end < 0` means "cover everything", the default every
        # existing caller hits, so the range is `[0, k_full_strips)` unless
        # a chained caller narrows it.
        var k_range_start = Int32(k_strip_start)
        var k_range_end = k_full_strips if k_strip_end < 0 else Int32(
            k_strip_end
        )

        # Pre-tile this simdgroup's SG_N-col block of B once (full K), so the
        # K-loop tiles only the K axis with the column offset hoisted out of the
        # hot loop -- same hoist as the A slab (see the `run` wrapper docstring).
        # `b_ptr_shifted`/`c_ptr_shifted`: same shift as the A slab in `run`
        # (no-op when `clamp_active` is False).
        var b_ptr_shifted = b_ptr + Int(col_shift) * (
            k if Self.transpose_b else 1
        )
        var c_ptr_shifted = c_ptr + (Int(row_shift) * n + Int(col_shift))
        var b_mat = TileTensor[linear_idx_type=Self.linear_idx_type](
            b_ptr_shifted, Self._pick_b_mat_layout(k, n)
        )
        var b_slab = b_mat.tile(Coord(k, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        # Pass 1 seeds from pass 0's stored output instead of zeroing. Safe
        # because `clamp_v2` ownership is a strict partition: every cell read
        # here was already written by pass 0, and a stray read of a
        # neighbor's cell gets discarded by this tile's own bounded store.
        comptime do_seed = seed_from_output and not (
            Self.c_type != .float32 or Self.elementwise_lambda_fn
        )
        var accum: Mma.AccumType
        comptime if do_seed:
            var c_ptr_fp32_seed = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                c_ptr_shifted
            )
            var c_mat_fp32_seed = TileTensor[
                linear_idx_type=Self.linear_idx_type
            ](c_ptr_fp32_seed, row_major(m, n))
            var c_sub_fp32_seed = c_mat_fp32_seed.tile[SG_M, SG_N](
                Int(sg_row_idx), Int(sg_col_idx)
            )
            accum = mma_op.load_accum(c_sub_fp32_seed)
        else:
            accum = Mma.zero_accum()

        # fp32 out with no fused lambda takes the fast `mma_op.store` path;
        # every other (c_type, lambda) combo flows through the epilogue below.
        comptime use_epilogue_path = (
            Self.c_type != .float32 or Self.elementwise_lambda_fn
        )

        # Cast-then-store epilogue (non-fp32 out and/or a fused
        # `elementwise_lambda_fn`). Writes through a `.tile`-derived simdgroup
        # view of C -- no pointer arithmetic. The lambda contract matches AMD's:
        # it receives `SIMD[c_type, width]` at absolute (row, col).
        @always_inline
        @__parameter
        def _apply_epilogue[
            bounded: Bool
        ](tile_row_base: Int, tile_col_base: Int):
            var c_sub = TileTensor[linear_idx_type=Self.linear_idx_type](
                c_ptr, row_major(m, n)
            ).tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            # 4 contiguous output cols = one SIMD unit. Element alignment only:
            # the row stride `n` is odd for odd N, so the default full-vector
            # alignment would fault -- override it on the vectorized store.
            comptime elem_align = align_of[Scalar[Self.c_type]]()
            var c_vec = c_sub.vectorize[1, 4]()

            @always_inline
            @__parameter
            def _write4(
                lrow: Int,
                lcol: Int,
                arow: Int,
                acol: Int,
                v_fp32: SIMD[.float32, 4],
            ):
                # `lrow,lcol`: coords inside the simdgroup tile (C store).
                # `arow,acol`: absolute coords (bounds + the lambda contract).
                var y = v_fp32.cast[Self.c_type]()
                comptime if Self.elementwise_lambda_fn:
                    comptime epilogue = Self.elementwise_lambda_fn.value()
                    comptime if bounded:
                        if acol + 3 < n:
                            epilogue[Self.c_type, 4, alignment=elem_align](
                                IndexList[2](arow, acol), y
                            )
                        else:
                            for e in range(min(4, n - acol)):
                                epilogue[Self.c_type, 1](
                                    IndexList[2](arow, acol + e),
                                    SIMD[Self.c_type, 1](y[e]),
                                )
                    else:
                        epilogue[Self.c_type, 4, alignment=elem_align](
                            IndexList[2](arow, acol), y
                        )
                else:
                    comptime if bounded:
                        if acol + 3 < n:
                            c_vec.store[alignment=elem_align](
                                Coord(lrow, lcol // 4), y
                            )
                        else:
                            for e in range(min(4, n - acol)):
                                c_sub.store[width=1, alignment=elem_align](
                                    Coord(lrow, lcol + e),
                                    SIMD[Self.c_type, 1](y[e]),
                                )
                    else:
                        c_vec.store[alignment=elem_align](
                            Coord(lrow, lcol // 4), y
                        )

            comptime for mi in range(Mma.num_m_mmas):
                comptime for ni in range(Mma.num_n_mmas):
                    var frag = accum[mi * Mma.num_n_mmas + ni]
                    var lcol = ni * 16 + Int(mma_op.cb)
                    var lrow = mi * 16 + Int(mma_op.rb)
                    var acol = tile_col_base + lcol
                    var arow = tile_row_base + lrow
                    comptime if bounded:
                        if arow < m:
                            _write4(
                                lrow,
                                lcol,
                                arow,
                                acol,
                                frag.slice[4, offset=0](),
                            )
                        if arow + 8 < m:
                            _write4(
                                lrow + 8,
                                lcol,
                                arow + 8,
                                acol,
                                frag.slice[4, offset=4](),
                            )
                    else:
                        _write4(
                            lrow, lcol, arow, acol, frag.slice[4, offset=0]()
                        )
                        _write4(
                            lrow + 8,
                            lcol,
                            arow + 8,
                            acol,
                            frag.slice[4, offset=4](),
                        )

        # `rebind` to fp32 so `mma_op.store{,_bounded}` typechecks; this branch
        # is only entered when `c_type == fp32` (use_epilogue_path is False),
        # so the rebind is a no-op at runtime.
        @always_inline
        @__parameter
        def _fast_path_store[
            bounded: Bool
        ](valid_rows: Int = 0, valid_cols: Int = 0):
            var c_ptr_fp32 = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                c_ptr_shifted
            )
            var c_mat_fp32 = TileTensor[linear_idx_type=Self.linear_idx_type](
                c_ptr_fp32, row_major(m, n)
            )
            var c_sub_fp32 = c_mat_fp32.tile[SG_M, SG_N](
                Int(sg_row_idx), Int(sg_col_idx)
            )
            comptime if bounded:
                mma_op.store_bounded(accum, c_sub_fp32, valid_rows, valid_cols)
            else:
                mma_op.store(accum, c_sub_fp32)

        # K-loop loads B directly from device memory each step; the A side comes
        # from the comptime loader (slab tile vs online im2col gather).
        # Threadgroup-memory staging *degrades* matmul on Apple Silicon.
        if is_edge_tile:
            # Belt-and-suspenders clamp. The early return guarantees
            # `[1, SG_M]` in steady state; this survives a future refactor
            # that drops it. Inside `MmaOpApple.mma`, `valid_* - mi*16`
            # may still go negative for partial tiles -- `_bounded_load`
            # zero-fills, which is correct.
            var valid_rows = max(Int32(1), min(SG_M_i32, m_i32 - row_base))
            var valid_cols = max(Int32(1), min(SG_N_i32, n_i32 - col_base))
            # Split the main loop (k_valid=BK, a compile-time constant) from
            # the tail, matching the non-edge path below. Keeps the M/N-edge
            # mask inputs loop-invariant instead of re-deriving them from
            # min() every strip.
            for k_strip in range(k_full_strips):
                var b_sub = b_slab.tile[BK, SG_N](Int(k_strip), 0)
                loader.accumulate_strip[bounded=True](
                    mma_op,
                    accum,
                    b_sub,
                    conv,
                    k_strip,
                    a_valid_rows=valid_rows,
                    b_valid_cols=Int(valid_cols),
                    k_valid=BK,
                )
            if has_k_tail:
                var k_tail = k_i32 - k_full_strips * BK_i32
                var b_sub = b_slab.tile[BK, SG_N](Int(k_full_strips), 0)
                loader.accumulate_strip[bounded=True](
                    mma_op,
                    accum,
                    b_sub,
                    conv,
                    k_full_strips,
                    a_valid_rows=valid_rows,
                    b_valid_cols=Int(valid_cols),
                    k_valid=Int(k_tail),
                )
            comptime if use_epilogue_path:
                _apply_epilogue[bounded=True](Int(row_base), Int(col_base))
            else:
                _fast_path_store[bounded=True](Int(valid_rows), Int(valid_cols))
        else:

            @always_inline
            @__parameter
            def _full_strip(k_strip: Int32):
                var b_sub = b_slab.tile[BK, SG_N](Int(k_strip), 0)
                loader.accumulate_strip[bounded=False](
                    mma_op,
                    accum,
                    b_sub,
                    conv,
                    k_strip,
                    a_valid_rows=SG_M_i32,
                    b_valid_cols=SG_N,
                    k_valid=BK,
                )

            comptime if Self.k_unroll > 1:
                comptime UNROLL_i32 = Int32(Self.k_unroll)
                var span = k_range_end - k_range_start
                var full_end = k_range_start + (span - span % UNROLL_i32)
                for g in range(
                    Int32(0), (full_end - k_range_start) // UNROLL_i32
                ):
                    var ks = k_range_start + g * UNROLL_i32
                    comptime for u in range(Self.k_unroll):
                        _full_strip(ks + Int32(u))
                for k_strip in range(full_end, k_range_end):
                    _full_strip(k_strip)
            else:
                for k_strip in range(k_range_start, k_range_end):
                    _full_strip(k_strip)
            # Only the pass reaching the true end of K handles the tail.
            if has_k_tail and k_range_end == k_full_strips:
                var k_tail = k_i32 - k_full_strips * BK_i32
                var b_sub = b_slab.tile[BK, SG_N](Int(k_full_strips), 0)
                loader.accumulate_strip[bounded=True](
                    mma_op,
                    accum,
                    b_sub,
                    conv,
                    k_full_strips,
                    a_valid_rows=SG_M_i32,
                    b_valid_cols=SG_N,
                    k_valid=Int(k_tail),
                )
            comptime if use_epilogue_path:
                _apply_epilogue[bounded=False](Int(row_base), Int(col_base))
            else:
                comptime if clamp_active:
                    _fast_path_store[bounded=True](
                        Int(v2_valid_rows), Int(v2_valid_cols)
                    )
                else:
                    _fast_path_store[bounded=False]()

    # === Single-pass kernel ================================================ #

    @__name(
        t"apple_matmul_run_{Self.in_type}_{Self.c_type}_tb{Self.transpose_b}_b{Self.b_type}"
    )
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """GEMM kernel entry; `M`/`N`/`K` derive from the operands.

        A is `(M, K)` row-major and C is `(M, N)` row-major; B is `(K, N)` for
        `transpose_b=False` or `(N, K)` for `transpose_b=True`. Grid is
        `(1<<log2_grid_m) * (1<<log2_grid_n)` threadgroups of 128 threads; OOB
        threadgroups early-return after Morton decode.

        Thin wrapper over `_run_gemm_body`: derives `K`, pre-tiles this
        simdgroup's `(SG_M, K)` A slab, and constructs the `DenseALoader` that
        reads it.

        Parameters:
            c_layout: `TensorLayout` of the output `C` operand.
            a_layout: `TensorLayout` of the A operand.
            b_layout: `TensorLayout` of the B operand.
            c_engine: `TensorEngine` of the output `C` operand.
            a_engine: `TensorEngine` of the A operand.
            b_engine: `TensorEngine` of the B operand.

        Args:
            c: Output matrix `(M, N)` row-major; `M` and `N` derive from its
                dims.
            a: A operand matrix `(M, K)` row-major; `K` derives from
                `a.dim[1]`.
            b: B operand matrix, `(K, N)` for `transpose_b=False` or `(N, K)`
                for `transpose_b=True`.
            log2_grid_m: Base-2 logarithm of the M-axis grid extent; the grid
                spans `1<<log2_grid_m` threadgroups along M.
            log2_grid_n: Base-2 logarithm of the N-axis grid extent; the grid
                spans `1<<log2_grid_n` threadgroups along N.
        """
        var m = Int(c.dim[0]())
        var k = Int(a.dim[1]())

        comptime SG_M = Self.SG_M

        # Pre-tile the SG_M-row A slab ONCE here, outside the K-loop: the
        # hot-loop `.tile[SG_M, BK](0, k_strip)` then only adds `k_strip * BK`,
        # keeping the invariant `sg_row * SG_M * K` base out of the K-loop (worth
        # ~+3-6% on large shapes; the B slab is hoisted the same way in the
        # body). The row base comes via `_sg_row_base` so it matches the body's
        # inline `row_base` bit-for-bit; OOB simdgroups build a slab they never
        # read (the body early-returns before the K-loop), so this is cheap and
        # side-effect-free. `UntrackedOrigin` so the slab can be a loader field
        # (struct fields cannot expose `AnyOrigin`); `a` outlives the body.
        var row_base = Self._sg_row_base(log2_grid_m, log2_grid_n)
        var sg_row_idx = row_base // Int32(SG_M)
        var a_ptr = a.ptr.unsafe_origin_cast[ImmUntrackedOrigin]()

        var a_mat = TileTensor[linear_idx_type=Self.linear_idx_type](
            a_ptr, Layout(Coord(m, k), Coord(k, Idx[1]))
        )
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))
        var loader = DenseALoader[
            Self.in_type,
            type_of(a_slab).LayoutType,
            BK=Self.BK,
            SG_M=Self.SG_M,
            use_x2=Self.use_x2,
            b_dtype=Self.b_type,
        ](a_slab)

        var no_conv = ConvIm2colParams()  # dense path ignores conv
        # B policy: direct-DRAM (dense-bf16 or FP8-W8A16), no SMEM.
        Self._run_gemm_body[W=DenseWeightLoader[Self.b_type, Self.accum_type]](
            loader, c, b, k, no_conv, log2_grid_m, log2_grid_n
        )

    # === Chained clamp_v2 kernel: 2-pass, no partials buffer =============== #

    @__name(
        t"apple_matmul_run_chained_{Self.in_type}_{Self.c_type}_tb{Self.transpose_b}_sfo{seed_from_output}"
    )
    @staticmethod
    def run_chained[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
        seed_from_output: Bool = False,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
        k_strip_start_dev: Int32,
        k_strip_end_dev: Int32,
    ):
        """`clamp_v2` chained-pass kernel. `enqueue_apple_matmul_clamp_chain`
        launches it twice: pass 0 zero-seeds `[k_strip_start, k_strip_end)`,
        pass 1 seeds from `c` and covers the rest, so no partials buffer or
        reduce kernel is needed. Otherwise identical to `run`; requires
        `Self.clamp_edge=True` (asserted below).
        """
        # `Int` is not device-passable; widen the fixed-width args.
        var k_strip_start = Int(k_strip_start_dev)
        var k_strip_end = Int(k_strip_end_dev)
        comptime assert (
            Self.clamp_edge
            and Self.c_type == .float32
            and not Self.elementwise_lambda_fn
        ), (
            "run_chained requires clamp_edge=True, c_type=float32, no"
            " elementwise_lambda_fn (see enqueue_apple_matmul_clamp_chain)"
        )

        var m = Int(c.dim[0]())
        var k = Int(a.dim[1]())

        comptime SG_M = Self.SG_M
        comptime BM_i32: Int32 = Int32(Self.BM)

        var row_base = Self._sg_row_base(log2_grid_m, log2_grid_n)
        var sg_row_idx = row_base // Int32(SG_M)
        var a_ptr = a.ptr.unsafe_origin_cast[ImmUntrackedOrigin]()

        # Same A-slab clamp shift as `run` (see its comment). Unconditional
        # here since the `comptime assert` above guarantees `clamp_edge=True`.
        var tile_row0 = align_down(row_base, BM_i32)
        if tile_row0 + BM_i32 > Int32(m):
            var row_shift = (Int32(m) - BM_i32) - tile_row0
            a_ptr = a_ptr + Int(row_shift) * k

        var a_mat = TileTensor[linear_idx_type=Self.linear_idx_type](
            a_ptr, Layout(Coord(m, k), Coord(k, Idx[1]))
        )
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))
        var loader = DenseALoader[
            Self.in_type,
            type_of(a_slab).LayoutType,
            BK=Self.BK,
            SG_M=Self.SG_M,
            use_x2=Self.use_x2,
            b_dtype=Self.b_type,
        ](a_slab)

        var no_conv = ConvIm2colParams()  # dense path ignores conv
        # B policy: direct-DRAM dense fp32 (clamp_v2 chained), no SMEM.
        Self._run_gemm_body[
            W=DenseWeightLoader[Self.b_type, Self.accum_type],
            seed_from_output=seed_from_output,
        ](
            loader,
            c,
            b,
            k,
            no_conv,
            log2_grid_m,
            log2_grid_n,
            k_strip_start,
            k_strip_end,
        )

    # === Fused online-im2col conv kernel =================================== #
    # Same simdgroup-tiled GEMM body as `run` (`_run_gemm_body`), but the A
    # operand is the conv im2col matrix `[M=N*H_out*W_out, K=R*S*C]` -- gathered
    # on the fly from the NHWC input per MMA-fragment instead of materialised to
    # global memory. Mirrors the MI355 conv pattern (swap only the A-operand
    # loader; share the matmul body); the Apple seam is the FRAGMENT LOAD, not an
    # LDS stage, because Apple matmul has no shared-memory staging (KB
    # `patterns/apple-m5-gpu-performance-considerations`,
    # `exceptions/apple-mma-fragment-is-not-distribute-expressible`). The K, M,
    # N decomposition and OOB zero-fill match `nn/conv/gpu/im2col_matmul_2d.mojo`
    # so results match the materialised path. B is the filter `[N=C_out, K]`
    # (transpose_b=True). bf16 only for now.

    @__name(t"apple_conv2d_run_{Self.in_type}_{Self.c_type}_ca{c_aligned}")
    @staticmethod
    def run_conv[
        c_layout: TensorLayout,
        input_layout: TensorLayout,
        b_layout: TensorLayout,
        c_engine: TensorEngine,
        input_engine: TensorEngine,
        b_engine: TensorEngine,
        c_aligned: Bool = False,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        input: TileTensor[
            Self.in_type, input_layout, ImmutAnyOrigin, Engine=input_engine
        ],
        # Conv filter is `in_type` (the `Im2colALoader`'s `b_type == in_type`);
        # the body's `b` param keys on the loader's `b_type`, so this stays
        # `Self.in_type`, not `Self.b_type`.
        b: TileTensor[Self.in_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        conv: ConvIm2colParams,
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """Fused conv2d GEMM entry; `M`/`N`/`K` derive from C and the conv params.

        C is `(M, N)` row-major with `M = N_batch*H_out*W_out`, `N = C_out`.
        `input` is the 4-D NHWC source (flat pointer used for the gather). B is
        the filter `(N, K)` with `transpose_b=True` (the NK layout the existing
        matmul path uses). Requires `transpose_b == True`. Grid is identical to
        `run`. The fused epilogue (if any) and the partial-edge bounds are the
        same as `run` -- only the A-fragment producer changes.

        Thin wrapper over `_run_gemm_body`: derives `K = R*S*C` and constructs an
        `Im2colALoader` over `input` (no slab -- the gather reads NHWC directly).
        `conv` is threaded to the body as a value arg (NOT held in the loader --
        see `Im2colALoader` for the Metal addrspace reason), then runs the body.

        Parameters:
            c_layout: `TensorLayout` of the output `C` operand.
            input_layout: `TensorLayout` of the NHWC `input` operand.
            b_layout: `TensorLayout` of the filter `B` operand.
            c_engine: `TensorEngine` of the output `C` operand.
            input_engine: `TensorEngine` of the NHWC `input` operand.
            b_engine: `TensorEngine` of the filter `B` operand.
            c_aligned: When True, assume `conv.C` is a multiple of 8 so the
                channel run is contiguous and a single width-8 load suffices
                (defaults to False).

        Args:
            c: Output matrix `(M, N)` row-major with `M = N_batch*H_out*W_out`
                and `N = C_out`.
            input: 4-D NHWC source tensor; its flat pointer drives the im2col
                gather.
            b: Filter operand `(N, K)` with `transpose_b=True` (NK layout).
            conv: Conv geometry for the im2col gather; threaded to the body as
                a value arg.
            log2_grid_m: Base-2 logarithm of the M-axis grid extent; the grid
                spans `1<<log2_grid_m` threadgroups along M.
            log2_grid_n: Base-2 logarithm of the N-axis grid extent; the grid
                spans `1<<log2_grid_n` threadgroups along N.
        """
        comptime assert (
            Self.transpose_b
        ), "run_conv requires transpose_b=True (filter NK layout)"

        var k = Int(conv.R) * Int(conv.S) * Int(conv.C)  # K = R*S*C_in

        # Prebake this lane's anchors + K-state ONCE here, outside the K-loop
        # (mirrors the dense slab prebake -- KB `kernels/apple-conv2d-im2col`).
        # `rb`/`cb` MUST use `MmaOpApple.__init__`'s lane formula and `row_base`
        # MUST match the body's inline `row_base`, or the anchors misalign with
        # the gather's fragment rows. OOB simdgroups prebake anchors they never
        # read (the body early-returns before the K-loop) -- harmless.
        var row_base = Self._sg_row_base(log2_grid_m, log2_grid_n)
        var lid = Int(lane_id())
        var rb = Int32(((lid & 7) >> 1) + ((lid & 16) >> 2))
        var cb = Int32(((lid & 1) << 2) + (lid & 8))
        # `UntrackedOrigin` so `input_ptr` can be a loader field (struct fields
        # cannot expose `AnyOrigin`); `input` outlives the body's gather.
        var loader = Im2colALoader[
            Self.in_type,
            Self.BK,
            Self.NUM_MMA_M,
            Self.NUM_MMA_N,
            c_aligned=c_aligned,
        ](
            input.ptr.unsafe_origin_cast[ImmUntrackedOrigin](),
            row_base,
            rb,
            cb,
            conv,
        )
        # Conv B (the filter) is `in_type`; direct-DRAM policy, no SMEM.
        Self._run_gemm_body[W=DenseWeightLoader[Self.b_type, Self.accum_type]](
            loader, c, b, k, conv, log2_grid_m, log2_grid_n
        )

    # === Split-K kernels =================================================== #
    # Partition the K axis across `num_splits` threadgroup-sets so large-K /
    # small-M*N shapes (few output tiles -> low occupancy) get more parallelism.
    # Each split writes an fp32 partial to a workspace; the reduce pass sums the
    # partials and applies the cast + epilogue. Deterministic -- no global
    # atomics (Apple fp32 atomic-add is not relied upon).

    @__name(t"apple_matmul_split_k_partial_{Self.in_type}_tb{Self.transpose_b}")
    @staticmethod
    def run_split_k_partial[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
    ](
        partials_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.in_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
        k_per_split: Int32,
    ):
        """One 64x64 output tile's fp32 partial over a BK-aligned K-slice.

        Grid is `(side_m * side_n) * num_splits` threadgroups. The split index
        selects the K-range `[s*k_per_split, min(K, (s+1)*k_per_split))` and the
        `partials[s]` output matrix; the tile index is Morton-decoded as in
        `run`. `k_per_split` is a multiple of `BK`, so every split but the last
        is full BK strips; the last may carry a partial-BK tail. No cast, no
        epilogue -- raw fp32 accumulator out.

        Parameters:
            a_layout: `TensorLayout` of the A operand.
            b_layout: `TensorLayout` of the B operand.
            a_engine: `TensorEngine` of the A operand.
            b_engine: `TensorEngine` of the B operand.

        Args:
            partials_ptr: Pointer to the fp32 partials workspace; split `s`
                writes its partial at offset `s * M * N`.
            a: A operand matrix `(M, K)` row-major; `M` and `K` derive from
                its dims.
            b: B operand matrix, `(K, N)` for `transpose_b=False` or `(N, K)`
                for `transpose_b=True`.
            log2_grid_m: Base-2 logarithm of the M-axis grid extent; the grid
                spans `1<<log2_grid_m` threadgroups along M.
            log2_grid_n: Base-2 logarithm of the N-axis grid extent; the grid
                spans `1<<log2_grid_n` threadgroups along N.
            k_per_split: K extent per split, a multiple of `BK`; split `s`
                owns `[s*k_per_split, min(K, (s+1)*k_per_split))`.
        """
        var a_ptr = a.ptr
        var b_ptr = b.ptr
        var m = Int(a.dim[0]())
        var k = Int(a.dim[1]())
        var n = Int(b.dim[0]()) if Self.transpose_b else Int(b.dim[1]())

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N
        # Split-K is the bf16-family, fp32-accumulate path: A and B are both
        # `in_type` and the partials are fp32, so pin the MMA to those (NOT the
        # generalized `Self.Mma`, whose `b_type`/`accum_type` are symbolic and do
        # not unify with the `in_type` operands / fp32 partials here). fp8 does
        # not route to split-K.
        comptime Mma = _BodyMma[Self.in_type, Self.NUM_MMA_M, Self.NUM_MMA_N]

        var m_i32 = Int32(m)
        var n_i32 = Int32(n)
        var k_i32 = Int32(k)
        comptime BM_i32: Int32 = Int32(BM)
        comptime BN_i32: Int32 = Int32(BN)
        comptime BK_i32: Int32 = Int32(BK)
        comptime SG_M_i32: Int32 = Int32(SG_M)
        comptime SG_N_i32: Int32 = Int32(SG_N)
        comptime NUM_SG_N_i32: Int32 = Int32(Self.NUM_SG_N)

        var grid_m = ceildiv(m_i32, BM_i32)
        var grid_n = ceildiv(n_i32, BN_i32)

        var num_tiles = UInt32(1) << (log2_grid_m + log2_grid_n)
        var bx = UInt32(block_idx.x)
        var split_idx = Int32(bx // num_tiles)
        var tile_flat = bx % num_tiles

        var tile_mn = Self.morton_decode_2d_rect(
            tile_flat, log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var k0 = split_idx * Int32(k_per_split)
        if k0 >= k_i32:
            return
        var k1 = min(k_i32, k0 + Int32(k_per_split))
        var strip0 = k0 // BK_i32
        var span = k1 - k0
        var full_strips = span // BK_i32
        var has_tail = (span % BK_i32) != 0

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // NUM_SG_N_i32
        var sg_n_idx = sg_id % NUM_SG_N_i32

        var mma_op = Mma()
        var accum = Mma.zero_accum()

        var row_base = tile_m * BM_i32 + sg_m_idx * SG_M_i32
        var col_base = tile_n * BN_i32 + sg_n_idx * SG_N_i32
        if row_base >= m_i32 or col_base >= n_i32:
            return
        var sg_row_idx = row_base // SG_M_i32
        var sg_col_idx = col_base // SG_N_i32
        var is_edge_tile = (row_base + SG_M_i32 > m_i32) or (
            col_base + SG_N_i32 > n_i32
        )

        var a_mat = TileTensor[linear_idx_type=Self.linear_idx_type](
            a_ptr, Layout(Coord(m, k), Coord(k, Idx[1]))
        )
        var b_mat = TileTensor[linear_idx_type=Self.linear_idx_type](
            b_ptr, Self._pick_b_mat_layout(k, n)
        )
        # Hoist the simdgroup base offset out of the K-loop (see `run`).
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))
        var b_slab = b_mat.tile(Coord(k, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        var total_strips = Int(full_strips) + (1 if has_tail else 0)
        if is_edge_tile:
            var valid_rows = max(Int32(1), min(SG_M_i32, m_i32 - row_base))
            var valid_cols = max(Int32(1), min(SG_N_i32, n_i32 - col_base))
            for j in range(total_strips):
                var gstrip = strip0 + Int32(j)
                var k_valid = min(BK_i32, k1 - gstrip * BK_i32)
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=Int(valid_rows),
                    b_valid_cols=Int(valid_cols),
                    k_valid=Int(k_valid),
                )
            var part_sub = TileTensor[linear_idx_type=Self.linear_idx_type](
                partials_ptr + Int(split_idx) * m * n, row_major(m, n)
            ).tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            mma_op.store_bounded(
                accum, part_sub, Int(valid_rows), Int(valid_cols)
            )
        else:

            @always_inline
            @__parameter
            def _full_strip(gstrip: Int32):
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                comptime if Self.use_x2:
                    mma_op.mma_dense_x2(accum, a_sub, b_sub)
                else:
                    mma_op.mma(accum, a_sub, b_sub)

            comptime if Self.k_unroll > 1:
                comptime UNROLL_i32 = Int32(Self.k_unroll)
                var full_end = full_strips - (full_strips % UNROLL_i32)
                for g in range(Int32(0), full_end // UNROLL_i32):
                    var base = g * UNROLL_i32
                    comptime for u in range(Self.k_unroll):
                        _full_strip(strip0 + base + Int32(u))
                for j in range(full_end, full_strips):
                    _full_strip(strip0 + j)
            else:
                for j in range(full_strips):
                    _full_strip(strip0 + j)
            if has_tail:
                var gstrip = strip0 + full_strips
                var k_valid = k1 - gstrip * BK_i32
                var a_sub = a_slab.tile[SG_M, BK](0, Int(gstrip))
                var b_sub = b_slab.tile[BK, SG_N](Int(gstrip), 0)
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=SG_M,
                    b_valid_cols=SG_N,
                    k_valid=Int(k_valid),
                )
            var part_sub = TileTensor[linear_idx_type=Self.linear_idx_type](
                partials_ptr + Int(split_idx) * m * n, row_major(m, n)
            ).tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            mma_op.store(accum, part_sub)

    @__name(t"apple_matmul_split_k_reduce_{Self.c_type}")
    @staticmethod
    def run_split_k_reduce[
        c_layout: TensorLayout,
        c_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        partials_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
        num_splits: Int32,
    ):
        """Sum `num_splits` fp32 partials per output element, cast, store / fuse.

        One thread per output element; `idx = block_idx.x * block_dim.x +
        thread_idx.x`. The fused `elementwise_lambda_fn` (if any) sees the
        absolute (row, col) and the final `SIMD[c_type, 1]`.

        Parameters:
            c_layout: `TensorLayout` of the output `C` operand.
            c_engine: `TensorEngine` of the output `C` operand.

        Args:
            c: Output matrix `(M, N)` row-major; `M` and `N` derive from its
                dims.
            partials_ptr: Pointer to the fp32 partials workspace; split `s`
                partial is at offset `s * M * N`.
            num_splits: Number of K splits to sum per output element.
        """
        var _num_splits = Int(num_splits)
        # `c_type` / `elementwise_lambda_fn` are struct params -- use `Self.x`.
        var c_ptr = c.ptr
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())

        var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        var total = m * n
        if idx >= total:
            return

        var acc = Float32(0)
        var mn = m * n
        for s in range(_num_splits):
            acc += partials_ptr[s * mn + idx]

        var y = acc.cast[Self.c_type]()
        comptime if Self.elementwise_lambda_fn:
            comptime epilogue = Self.elementwise_lambda_fn.value()
            epilogue[Self.c_type, 1](
                IndexList[2](idx // n, idx % n), SIMD[Self.c_type, 1](y)
            )
        else:
            c_ptr[idx] = y


# === Host-side launchers (standalone for testing) ========================== #


@always_inline
def enqueue_apple_matmul[
    in_type: DType,
    c_type: DType = .float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    b: TileTensor[in_type, ...],
    ctx: DeviceContext,
    force_split_k: Optional[Bool] = None,
) raises:
    """Enqueue `AppleM5MatMul.run` on the given device context.

    Accepts row-major TileTensor operands. For `transpose_b=True`, B is expected
    with shape `(N, K)`.

    `force_split_k` picks the K-reduction strategy: `None` (default) auto-routes
    under-occupied shapes (few output tiles, deep K) to split-K; `True` always
    uses split-K; `False` always uses the single-pass kernel.

    Parameters:
        in_type: A/B element type (fp16, bf16, fp32).
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32
            (defaults to `float32`).
        transpose_b: If True, B is `(N, K)` row-major (viewed as
            `col_major(K, N)`); otherwise B is `(K, N)` row-major (defaults to
            False).
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract)
            (defaults to None).

    Args:
        c: Output matrix `(M, N)` row-major; `M` and `N` derive from its dims.
        a: A operand matrix `(M, K)` row-major; `K` derives from `a.dim[1]`.
        b: B operand matrix, `(K, N)` for `transpose_b=False` or `(N, K)` for
            `transpose_b=True`.
        ctx: Device context to enqueue the kernel on; must be Apple M5
            (`compute_capability == 5`).
        force_split_k: K-reduction strategy override; `None` auto-routes
            under-occupied shapes to split-K, `True` always uses split-K,
            `False` always uses the single-pass kernel (defaults to None).

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
        M1-M4 lack GPU `neural accelerator`; future generations require
        re-validation.
    """
    # use_x2 needs transpose_b=True (K-contiguous B); k_unroll is the NN
    # equivalent win and the two regress badly combined (see AppleM5MatMul's
    # param docs), so they're mutually exclusive here. fp32 gets neither
    # (unvalidated).
    comptime use_x2 = transpose_b and (
        in_type == .bfloat16 or in_type == .float16
    )
    comptime block_k = 32 if use_x2 else 16
    comptime k_unroll = 4 if (
        not use_x2 and (in_type == .bfloat16 or in_type == .float16)
    ) else 1

    comptime MM = AppleM5MatMul[
        in_type,
        c_type,
        transpose_b,
        elementwise_lambda_fn,
        block_k=block_k,
        use_x2=use_x2,
        k_unroll=k_unroll,
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_apple_matmul requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
            (
                ". Route M1-M4 to the naive matmul path; re-validate for"
                " future generations."
            ),
        )

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_apple_matmul: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(
        Int(c.dim[0]()) == m and Int(c.dim[1]()) == n, "C shape (M, N)"
    )
    comptime if transpose_b:
        debug_assert(
            Int(b.dim[0]()) == n, "transpose_b=True expects B shape (N, K)"
        )
        debug_assert(
            Int(b.dim[1]()) == k, "transpose_b=True expects B shape (N, K)"
        )
    else:
        debug_assert(
            Int(b.dim[0]()) == k, "transpose_b=False expects B shape (K, N)"
        )
        debug_assert(
            Int(b.dim[1]()) == n, "transpose_b=False expects B shape (K, N)"
        )

    # MmaOpApple narrows the row stride to UInt16 (see `_load_fragment` /
    # `_store_fragment` in linalg/arch/apple/mma.mojo). Catch the wrap here:
    # NN A-slab stride = K, NN B-slab stride = N; NT B-slab stride = K (covered).
    debug_assert(k <= 65535, "Apple matmul: K must fit in UInt16; got K=", k)
    comptime if not transpose_b:
        debug_assert(
            n <= 65535, "Apple matmul (NN): N must fit in UInt16; got N=", n
        )

    # Per-axis next-pow2 grid for rectangular Z-order. e.g. 32x224 (Llama-3
    # MLP up-proj) -> 32x256 = 8192 launches vs the prior square 256x256 = 65536.
    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    # Split-K routing. `force_split_k` overrides the heuristic; when unset, the
    # heuristic routes under-occupied shapes -- few 64x64 output tiles but deep
    # K -- to split-K. The single-pass kernel launches one threadgroup per tile,
    # so a tiny M*N with large K leaves most of the GPU idle; splitting K
    # recovers 1.4-2.9x there (measured, M5 Max). The threshold is conservative;
    # normal shapes (many tiles) take the single-pass launch below.
    var tiles = grid_m * grid_n
    var num_strips = ceildiv(k, MM.BK)
    var route_split_k = force_split_k.value() if force_split_k else (
        tiles <= 16 and num_strips >= 32 and num_strips >= 8 * tiles
    )
    if route_split_k:
        var hint = max(2, min(num_strips // 4, 64 // tiles))
        enqueue_apple_matmul_split_k[
            in_type=in_type,
            c_type=c_type,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, ctx, hint)
        return

    # clamp+chain route: `clamp_v2` wins on ragged (M or N not a tile
    # multiple) dense NN bf16->fp32 shapes that don't already hit split-K.
    # Scope matches what's validated: K tile-aligned (no K-tail support) and
    # m >= BM, n >= BN (need a full tile to shift into). Anything else falls
    # through to the bounded-edge kernel below.
    #
    # Gated at comptime, not just the runtime check below: `run_chained`
    # asserts `c_type=float32` with no lambda, so without this `comptime if`
    # it would still get monomorphized (and fail that assert) for every
    # `enqueue_apple_matmul` instantiation, e.g. `c_type=float16`.
    comptime clamp_chain_dtype_ok = (
        in_type == .bfloat16
        and c_type == .float32
        and not elementwise_lambda_fn
        and not transpose_b
    )
    comptime if clamp_chain_dtype_ok:
        var route_clamp_chain = (
            (m % MM.BM != 0 or n % MM.BN != 0)
            and k % MM.BK == 0
            and m >= MM.BM
            and n >= MM.BN
        )
        if route_clamp_chain:
            enqueue_apple_matmul_clamp_chain[in_type=in_type, c_type=c_type](
                c, a, b, ctx
            )
            return

    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    var grid_dim = side_m * side_n

    # int32 offsets are only correct if every A/B/C tile offset fits int32; gate
    # on BYTE extent (element count * dtype size), the conservative bound.
    comptime a_bytes = size_of[Scalar[in_type]]()
    comptime c_bytes = size_of[Scalar[c_type]]()
    comptime kI32Max = Int(Int32.MAX)
    var fits_i32 = (
        (m * n) * c_bytes <= kI32Max
        and (m * k) * a_bytes <= kI32Max
        and (n * k) * a_bytes <= kI32Max
    )

    comptime kernel_i64 = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
    ]
    comptime MM_i32 = AppleM5MatMul[
        in_type,
        c_type,
        transpose_b,
        elementwise_lambda_fn,
        block_k=block_k,
        use_x2=use_x2,
        k_unroll=k_unroll,
        linear_idx_type=.int32,
    ]
    comptime kernel_i32 = MM_i32.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
    ]
    if fits_i32:
        ctx.enqueue_function[kernel_i32](
            c,
            a,
            b,
            log2_m,
            log2_n,
            grid_dim=(grid_dim),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
        return
    ctx.enqueue_function[kernel_i64](
        c,
        a,
        b,
        log2_m,
        log2_n,
        grid_dim=(grid_dim),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


@always_inline
def enqueue_apple_conv2d[
    in_type: DType,
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    input: TileTensor[in_type, ...],
    filter_nk: TileTensor[in_type, ...],
    conv: ConvIm2colParams,
    ctx: DeviceContext,
) raises:
    """Enqueue the fused online-im2col conv2d (`AppleM5MatMul.run_conv`).

    SM100/Apple M5 (`compute_capability == 5`). No `[M, K]` scratch is
    materialised: the A operand is gathered from `input` (4-D NHWC) on the fly.
    `filter_nk` is the filter pre-transposed to `(C_out, K=R*S*C_in)` row-major
    (the NK layout `dispatch_im2col_matmul_conv2d` already builds); the GEMM uses
    `transpose_b=True`. C is `(M=N_batch*H_out*W_out, N=C_out)` row-major (a flat
    view of the NHWC output). Grid mirrors `enqueue_apple_matmul` (single-pass;
    no split-K for conv yet).

    Parameters:
        in_type: Element type of the NHWC `input` and `filter_nk` operands (bf16
            for now).
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32
            (defaults to `float32`).
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract)
            (defaults to None).

    Args:
        c: Output matrix `(M, N)` row-major with `M = N_batch*H_out*W_out` and
            `N = C_out` (a flat view of the NHWC output).
        input: 4-D NHWC source tensor; its flat pointer drives the im2col
            gather.
        filter_nk: Filter pre-transposed to `(C_out, K=R*S*C_in)` row-major
            (the NK layout); used with `transpose_b=True`.
        conv: Conv geometry for the im2col gather.
        ctx: Device context to enqueue the kernel on; must be Apple M5
            (`compute_capability == 5`).

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
    """
    # Conv tile config: differs from the dense GEMM only in BK=32 (vs 16) for
    # the width-8 gather. NB: keep sg_m=sg_n=32 -- sg=16 (1 fragment) measured
    # ~1.7-1.9x SLOWER. Full config sweep: KB `kernels/apple-conv2d-im2col`.
    comptime MM = AppleM5MatMul[
        in_type,
        c_type,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        block_m=64,
        block_n=64,
        block_k=32,
        sg_m=32,
        sg_n=32,
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_apple_conv2d requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
        )

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(conv.R) * Int(conv.S) * Int(conv.C)

    # MmaOpApple narrows the B row stride to UInt16; for NT the B-slab stride is
    # K (col-major view), so K must fit. M is the gathered output-pixel count;
    # it never indexes a narrowed stride (A has no slab), so no M cap.
    debug_assert(
        k <= 65535, "Apple conv: K=R*S*C must fit in UInt16; got K=", k
    )
    debug_assert(
        Int(filter_nk.dim[0]()) == n,
        "filter_nk must be (C_out, K); C_out mismatch",
    )
    debug_assert(
        Int(filter_nk.dim[1]()) == k, "filter_nk must be (C_out, K); K mismatch"
    )
    # The online-im2col gather forms NHWC addresses in Int32 (batch*H*W*C +
    # h_in*W*C + w_in*C + c, plus the width-8 channel run). Cap the input extent
    # so that arithmetic can't silently wrap into a wrong/OOB load -- e.g.
    # batch>=8 at 1024^2/C256 exceeds 2^31. NOTE: the fix if that workload
    # appears is Int64 gather indices; we cap here rather than pay 64-bit ALU on
    # the hot path. `Int` is 64-bit so this product itself can't overflow.
    var in_elems = (
        Int(input.dim[0]())
        * Int(input.dim[1]())
        * Int(input.dim[2]())
        * Int(input.dim[3]())
    )
    debug_assert(
        in_elems <= 2147483647,
        "Apple conv: input N*H*W*C exceeds the Int32 gather-index range; got ",
        in_elems,
    )

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    var grid_dim = side_m * side_n

    # `C` is a runtime conv field, so this launch is the only place its
    # alignment can be lifted to a comptime kernel parameter without a per-shape
    # recompile. `c_aligned` lets the kernel DCE the per-element slow gather on
    # the interior strips (see `_load_a_im2col_fragment_x2`).
    @__parameter
    def _launch[c_aligned: Bool]() raises:
        comptime kernel = MM.run_conv[
            type_of(c).LayoutType,
            type_of(input).LayoutType,
            type_of(filter_nk).LayoutType,
            type_of(c).Engine,
            type_of(input).Engine,
            type_of(filter_nk).Engine,
            c_aligned=c_aligned,
        ]
        ctx.enqueue_function[kernel](
            c,
            input.as_immut(),
            filter_nk.as_immut(),
            conv,
            log2_m,
            log2_n,
            grid_dim=(grid_dim),
            block_dim=(MM.THREADS_PER_BLOCK),
        )

    if Int(conv.C) % 8 == 0:
        _launch[True]()
    else:
        _launch[False]()


@always_inline
def enqueue_apple_matmul_split_k[
    in_type: DType,
    c_type: DType = .float32,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    b: TileTensor[in_type, ...],
    ctx: DeviceContext,
    num_splits_hint: Int = 4,
) raises:
    """Split-K Apple M5 matmul: partition K, accumulate partials, reduce.

    `num_splits_hint` is an upper bound; the actual split count is capped so no
    split is empty (`actual_splits = ceil(num_strips / strips_per_split)` where
    `strips_per_split = ceil(num_strips / num_splits_hint)`). Best for large-K,
    small-M*N shapes where the single-pass kernel under-occupies the GPU.

    Parameters:
        in_type: A/B element type (fp16, bf16, fp32).
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32
            (defaults to `float32`).
        transpose_b: If True, B is `(N, K)` row-major (viewed as
            `col_major(K, N)`); otherwise B is `(K, N)` row-major (defaults to
            False).
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract)
            (defaults to None).

    Args:
        c: Output matrix `(M, N)` row-major; `M` and `N` derive from its dims.
        a: A operand matrix `(M, K)` row-major; `K` derives from `a.dim[1]`.
        b: B operand matrix, `(K, N)` for `transpose_b=False` or `(N, K)` for
            `transpose_b=True`.
        ctx: Device context to enqueue the kernels on; must be Apple M5
            (`compute_capability == 5`).
        num_splits_hint: Upper bound on the K-axis split count; the actual
            split count is capped so no split is empty (defaults to 4).

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability != 5`).
    """
    # Same NT-x2 / NN-k_unroll dispatch as enqueue_apple_matmul, so split-K
    # gets the same measured wins as the single-pass kernel instead of
    # comparing an optimized kernel to an unoptimized one.
    comptime use_x2 = transpose_b and (
        in_type == .bfloat16 or in_type == .float16
    )
    comptime block_k = 32 if use_x2 else 16
    comptime k_unroll = 4 if (
        not use_x2 and (in_type == .bfloat16 or in_type == .float16)
    ) else 1

    comptime MM = AppleM5MatMul[
        in_type,
        c_type,
        transpose_b,
        elementwise_lambda_fn,
        block_k=block_k,
        use_x2=use_x2,
        k_unroll=k_unroll,
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_apple_matmul_split_k requires Apple M5"
                " (compute_capability == 5); got "
            ),
            cc,
        )

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(k <= 65535, "Apple matmul: K must fit in UInt16; got K=", k)
    comptime if not transpose_b:
        debug_assert(
            n <= 65535, "Apple matmul (NN): N must fit in UInt16; got N=", n
        )

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)
    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    # Cap splits so none is empty: distribute BK-strips evenly (round up), then
    # recompute how many splits that actually fills.
    var num_strips = ceildiv(k, MM.BK)
    var hint = max(1, num_splits_hint)
    var strips_per_split = ceildiv(num_strips, hint)
    var actual_splits = ceildiv(num_strips, strips_per_split)
    var k_per_split = strips_per_split * MM.BK

    var partials = ctx.enqueue_create_buffer[.float32](actual_splits * m * n)

    # int32 offsets are only correct if every A/B/partials tile offset fits
    # int32; gate on byte extent (mirrors enqueue_apple_matmul's guard).
    # Each split's partial view only ever spans (m, n): the split offset is
    # applied to the raw 64-bit pointer before the int32-indexed view is
    # built.
    comptime a_bytes = size_of[Scalar[in_type]]()
    comptime kI32Max = Int(Int32.MAX)
    var fits_i32 = (
        (m * n) * size_of[Float32]() <= kI32Max
        and (m * k) * a_bytes <= kI32Max
        and (n * k) * a_bytes <= kI32Max
    )

    comptime MM_i32 = AppleM5MatMul[
        in_type,
        c_type,
        transpose_b,
        elementwise_lambda_fn,
        block_k=block_k,
        use_x2=use_x2,
        k_unroll=k_unroll,
        linear_idx_type=.int32,
    ]
    comptime partial_kernel_i64 = MM.run_split_k_partial[
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a).Engine,
        type_of(b).Engine,
    ]
    comptime partial_kernel_i32 = MM_i32.run_split_k_partial[
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a).Engine,
        type_of(b).Engine,
    ]
    if fits_i32:
        ctx.enqueue_function[partial_kernel_i32](
            partials.unsafe_ptr(),
            a,
            b,
            log2_m,
            log2_n,
            Int32(k_per_split),
            grid_dim=(side_m * side_n * actual_splits),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
    else:
        ctx.enqueue_function[partial_kernel_i64](
            partials.unsafe_ptr(),
            a,
            b,
            log2_m,
            log2_n,
            Int32(k_per_split),
            grid_dim=(side_m * side_n * actual_splits),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
    comptime reduce_kernel = MM.run_split_k_reduce[
        type_of(c).LayoutType, type_of(c).Engine
    ]
    var n_elems = m * n
    ctx.enqueue_function[reduce_kernel](
        c,
        partials.unsafe_ptr(),
        Int32(actual_splits),
        grid_dim=(ceildiv(n_elems, MM.REDUCE_BLOCK)),
        block_dim=(MM.REDUCE_BLOCK),
    )
    # Keep the workspace alive until both launches are enqueued.
    _ = partials^


@always_inline
def enqueue_apple_matmul_clamp_chain[
    in_type: DType,
    c_type: DType = .float32,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[in_type, ...],
    b: TileTensor[in_type, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the `clamp_v2` ragged-edge, 2-pass chained-accumulate dense
    GEMM (NN only). Pass 0 zero-seeds and covers the first half of K's
    strips; pass 1 seeds from `c` and covers the rest, overwriting `c` with
    the final result. No partials buffer, no separate reduce kernel.

    Caller guarantees `c_type == float32`, no `elementwise_lambda_fn`,
    `transpose_b == False`, `m >= BM`, `n >= BN`, and `k % block_k == 0`.
    Re-asserted here (`debug_assert`) since violating them walks the
    shifted A/B/C pointers out of their buffers' allocations.
    """
    comptime block_k = 16
    comptime k_unroll = 4 if (
        in_type == .bfloat16 or in_type == .float16
    ) else 1

    comptime MM = AppleM5MatMul[
        in_type,
        c_type,
        False,
        None,
        block_k=block_k,
        k_unroll=k_unroll,
        clamp_edge=True,
    ]

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(
        k % MM.BK == 0 and m >= MM.BM and n >= MM.BN,
        (
            "clamp+chain requires K tile-aligned and m>=BM,n>=BN (M/N may be"
            " ragged)"
        ),
    )

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)
    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1
    var grid_dim = side_m * side_n

    # Exact: caller guarantees k % block_k == 0, so there's no K-tail.
    var num_strips = k // MM.BK
    var strips_pass0 = num_strips // 2

    comptime a_bytes = size_of[Scalar[in_type]]()
    comptime c_bytes = size_of[Scalar[c_type]]()
    comptime kI32Max = Int(Int32.MAX)
    var fits_i32 = (
        (m * n) * c_bytes <= kI32Max
        and (m * k) * a_bytes <= kI32Max
        and (n * k) * a_bytes <= kI32Max
    )

    comptime MM_i32 = AppleM5MatMul[
        in_type,
        c_type,
        False,
        None,
        block_k=block_k,
        k_unroll=k_unroll,
        clamp_edge=True,
        linear_idx_type=.int32,
    ]

    comptime pass0_i64 = MM.run_chained[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        seed_from_output=False,
    ]
    comptime pass1_i64 = MM.run_chained[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        seed_from_output=True,
    ]
    comptime pass0_i32 = MM_i32.run_chained[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        seed_from_output=False,
    ]
    comptime pass1_i32 = MM_i32.run_chained[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        seed_from_output=True,
    ]

    if fits_i32:
        ctx.enqueue_function[pass0_i32](
            c,
            a,
            b,
            log2_m,
            log2_n,
            Int32(0),
            Int32(strips_pass0),
            grid_dim=(grid_dim),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
        ctx.enqueue_function[pass1_i32](
            c,
            a,
            b,
            log2_m,
            log2_n,
            Int32(strips_pass0),
            Int32(num_strips),
            grid_dim=(grid_dim),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
        return
    ctx.enqueue_function[pass0_i64](
        c,
        a,
        b,
        log2_m,
        log2_n,
        Int32(0),
        Int32(strips_pass0),
        grid_dim=(grid_dim),
        block_dim=(MM.THREADS_PER_BLOCK),
    )
    ctx.enqueue_function[pass1_i64](
        c,
        a,
        b,
        log2_m,
        log2_n,
        Int32(strips_pass0),
        Int32(num_strips),
        grid_dim=(grid_dim),
        block_dim=(MM.THREADS_PER_BLOCK),
    )
