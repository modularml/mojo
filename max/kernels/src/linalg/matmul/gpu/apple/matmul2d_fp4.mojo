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
"""Apple M5 weight-only NVFP4 (W4A16) matmul on the 16x32x16 `matmul2d` tile.

Target: Apple M5 (`compute_capability == 5`) only.

`out = activation @ weight^T` where the FP4 weight stays PACKED in DRAM
(`uint8 [N, K//2]` lo-nibble first + `float8_e4m3fn [N, K//16]` block scales).
This is the 16x32x16 `matmul2d`-tile sibling of the committed `AppleM5Fp4MatMul`
(which runs on the narrower `simdgroup_matrix` 16x16x16 MMA). Both paths coexist;
this one uses the wider 16x32x16 tile with the coalesced `transpose_right=1` NT
feed. "matmul2d" here names the 16x32x16 tile SHAPE (Apple's term), implemented
natively as two `air.simdgroup_matrix_16x16x16` MMAs -- one per 16-wide N half --
via the `_mma_apple_transposable` stdlib intrinsic (`matmul2d_mma_regc_bt_native`).
Pure Mojo over `_mma_apple`: no external library or codegen dependency, so it
builds on a stock toolchain.

## Why this is fast: the coalesced NT decode

The FP4 weight is the RIGHT operand with `transpose_b=True` (W is `(N, K)`).
On the base `matmul2d` swizzle an NT B fragment is a stride-K scattered gather
(31 TF/s). The `transpose_right=1` op reinterprets the right operand as `(N, K)`
so the B-fragment's fast axis is K -- a coalesced, K-contiguous load
(`bt_frag_coord`, below). For each B fragment this kernel decodes the packed FP4
weight DIRECTLY into the register fragment, K-contiguous:

- Per lane, the `transpose_right` B fragment reads 4 N-rows
  (`rb, rb+8, rb+16, rb+24`), each contributing 4 CONTIGUOUS K values
  (`cb, cb+1, cb+2, cb+3`, `cb` a multiple of 4). Those 4 K nibbles are exactly
  ONE 2-byte packed load (`packed[n, (k0+cb)//2]` as `uint16`), and (because
  `cb+3 < 16`) share ONE FP8 block scale (`scales[n, (k0+cb)//16]`).
- Decode is the FTZ-safe branch-free `decode_e2m1_to_f32` (bit-identical to the
  `E2M1_TO_FLOAT32` LUT the materialize oracle uses; the exponent-injection trick
  is WRONG on M5's flush-to-zero -- see `fp4_utils.mojo`), scaled by
  `|block_scale|` in f32, cast to bf16 -> the B fragment. So this path is
  BIT-EXACT vs the materialize -> dense oracle (same dequant arithmetic, same
  bf16 `matmul2d` MMA).

No SMEM, no barrier (Apple's idiom): DRAM packed weight decoded straight to the
register B fragment, MLX `NAXTile` register-C accumulators (`tm=tn=2`, 4
accumulators -- the M5 register-cliff optimum, do NOT exceed). Interior-only fast
path; the launcher rounds the grid up and the store guards `< M`/`< N`, so the
last partial tile writes only in-bounds. The packed-weight and scale loads on the
last K strip read up to the tile boundary; callers pass tile-aligned N and
K % 16 == 0 (the W4A16 production shapes are aligned; a bounded edge/tail variant
mirrors `AppleM5Fp4MatMul` if needed).

The NVFP4 per-tensor `weight_scale_2` scalar folds in OUTSIDE the kernel (a
post-matmul multiply by the graph lowering), identically to the committed path.
"""

from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from std.math import ceildiv
from max.gpu.sync import barrier
from max.gpu.compute.arch.mma_apple import _mma_apple_transposable
from max.gpu.host import DeviceContext
from std.memory import unsafe_stack_allocation
from std.utils import IndexList

from layout import Idx, TensorEngine, TileTensor
from layout.coord import Coord
from layout.tile_layout import Layout, TensorLayout

from linalg.fp4_utils import decode_e2m1_to_f32, NVFP4_SF_VECTOR_SIZE
from linalg.utils import elementwise_epilogue_type


# ===----------------------------------------------------------------------=== #
# The native 16x32x16 `matmul2d` MMA + its tile->lane fragment swizzle
# ===----------------------------------------------------------------------=== #
#
# The 16x32x16 `matmul2d` tile is implemented as two native
# `air.simdgroup_matrix_16x16x16_multiply_accumulate` calls (the
# `_mma_apple_transposable` stdlib intrinsic), one per 16-wide N half -- pure
# Mojo over `_mma_apple`, no external library or codegen dependency. The
# fragment swizzle (`*_frag_coord`) is the per-lane element map for the 16x32
# tile, matched bit-for-bit against a host oracle so the register load/store
# coordinates are exact.

# The `matmul2d` MMA tile shape (the b16_b16_f32 variant).
comptime M2D_M = 16
comptime M2D_N = 32
comptime M2D_K = 16


def _require_apple_m5(ctx: DeviceContext) raises:
    """Runtime guard: reject a non-Apple-M5 GPU at the host enqueue entry.

    The kernel body uses the Apple simdgroup MMA (`_mma_apple`), which compiles
    for any Apple GPU target, so the Metal-4 / M5 requirement is enforced at
    launch rather than with a compile-time assert. A compile-time assert would
    break a non-M5 Apple build (e.g. CI compiling for a non-M5 target); this
    matches the dense `enqueue_apple_matmul` pattern (runtime check, not a
    comptime assert).
    """
    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "matmul2d W4A16 (Apple M5 NVFP4) requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
        )


@always_inline
def frag_row_base(lane: Int) -> Int:
    """Base MMA-tile row for `lane` (before the per-element row jump).

    Args:
        lane: Simdgroup lane index in `[0, 32)`; determines the lane's base
            row in the 16x16 MMA tile.
    """
    var qid = lane >> 2
    return (qid & 4) | ((lane >> 1) & 3)


@always_inline
def frag_col_base(lane: Int) -> Int:
    """Base MMA-tile column for `lane` (before the per-element col offset).

    Args:
        lane: Simdgroup lane index in `[0, 32)`; determines the lane's base
            column in the 16x16 MMA tile.
    """
    var qid = lane >> 2
    return ((qid & 2) | (lane & 1)) * 4


@always_inline
def a_frag_coord(lane: Int, i: Int) -> IndexList[2]:
    """(row, col) in the 16x16 A tile for this lane's A element `i` (0..7).

    Args:
        lane: Simdgroup lane index in `[0, 32)`; determines the lane's base
            `(row, col)` in the 16x16 A tile.
        i: A-fragment element index in `[0, 8)`; selects which of this lane's
            8 held A elements to map to `(row, col)`.
    """
    return IndexList[2](
        frag_row_base(lane) + (i >> 2) * 8, frag_col_base(lane) + (i % 4)
    )


@always_inline
def bc_frag_coord(lane: Int, i: Int) -> IndexList[2]:
    """(row, col) in the 16x32 B/C tile for this lane's element `i` (0..15).

    Args:
        lane: Simdgroup lane index in `[0, 32)`; determines the lane's base
            `(row, col)` in the 16x32 B/C tile.
        i: B/C-fragment element index in `[0, 16)`; selects which of this
            lane's 16 held B/C elements to map to `(row, col)`.
    """
    var half = i // 8
    var sub = i % 8
    return IndexList[2](
        frag_row_base(lane) + (sub >> 2) * 8,
        frag_col_base(lane) + (sub % 4) + half * 16,
    )


@always_inline
def bt_frag_coord(lane: Int, i: Int) -> IndexList[2]:
    """(n, k) in the 16x32 right operand for element `i` under `transpose_right=1`.

    Under `transpose_right=1` the op reads the right operand as `(N, K)` instead
    of `(K, N)`. The B-fragment lane->element map is the SWAP of the base
    `bc_frag_coord`: the base row-axis (`frag_row_base + (sub>>2)*8`, plus
    `half*16`) carries N, and the base col-axis (`frag_col_base + (sub%4)`)
    carries K. So consecutive elements walk K CONTIGUOUSLY, which is stride-1 for
    an `(N, K)` weight -> a coalesced load (the fix for the stride-K scattered NT
    B gather). The C-store map is UNCHANGED (`bc_frag_coord`); transpose_right
    permutes only the right operand, not C. Matched bit-exact against a host
    oracle.

    Args:
        lane: Simdgroup lane index in `[0, 32)`; determines the lane's base
            `(n, k)` in the 16x32 right operand.
        i: B-fragment element index in `[0, 16)`; selects which of this lane's
            16 held B elements to map to `(n, k)` under `transpose_right=1`.
    """
    var half = i // 8
    var sub = i % 8
    return IndexList[2](
        frag_row_base(lane) + (sub >> 2) * 8 + half * 16,  # n (0..31)
        frag_col_base(lane) + (sub % 4),  # k (0..15, contiguous)
    )


@always_inline
def matmul2d_mma_regc_bt_native(
    a_frag: SIMD[.bfloat16, 8],
    b_frag: SIMD[.bfloat16, 16],
    mut c_acc: SIMD[.float32, 16],
):
    """Pure-Mojo `transpose_right=1` 16x32x16 MMA (native `simdgroup_matrix`).

    Fragment ABI: this lane's 8-element A, 16-element `transpose_right` B, and
    16-element f32 C accumulator; `(N, K)` right-operand layout; fp32 reduction.
    Calls the native `_mma_apple_transposable` intrinsic
    (`llvm.air.simdgroup_matrix_16x16x16_multiply_accumulate`) directly -- no
    external library or codegen dependency.

    The 16x32 tile decomposes into two side-by-side 16x16x16 MMAs along N:
    - half 0 (N 0..15):  A @ B[0:8]  -> C[0:8]
    - half 1 (N 16..31): A @ B[8:16] -> C[8:16]
    Each half's fragments are in the native `_apple_frag_layout` order: the
    per-lane element map (`bt_frag_coord` / `bc_frag_coord`) restricted to one
    N-half is bit-for-bit the native 16x16 fragment layout (`frag_row_base` /
    `frag_col_base` == `_apple_frag_layout` for all 32 lanes; the two C halves
    stitch back into exactly `bc_frag_coord`, so the C-store map is UNCHANGED).
    `transpose_a=False, transpose_b=True` reads B as `(N, K)`, matching the
    coalesced `bt_frag_coord` feed.

    Args:
        a_frag: This lane's 8-element A fragment (16x16 tile).
        b_frag: This lane's 16-element B fragment (op transpose_right layout;
            elements 0-7 = N 0..15, 8-15 = N 16..31).
        c_acc: This lane's 16-element C accumulator, updated in place.
    """
    var d_lo = SIMD[.float32, 8](0)
    _mma_apple_transposable(
        d_lo,
        a_frag,
        b_frag.slice[8, offset=0](),
        c_acc.slice[8, offset=0](),
        False,
        True,
    )
    var d_hi = SIMD[.float32, 8](0)
    _mma_apple_transposable(
        d_hi,
        a_frag,
        b_frag.slice[8, offset=8](),
        c_acc.slice[8, offset=8](),
        False,
        True,
    )
    c_acc = d_lo.join(d_hi)


# ===----------------------------------------------------------------------=== #
# The FP4 weight decode loader (TileIO expert-object, Apple analog)
# ===----------------------------------------------------------------------=== #
#
# Every DRAM <-> SMEM <-> register transition in this kernel has an OWNER (KB
# `new-primitives/amd-tile-io-expert-objects`; the Apple sibling of the AMD
# `TileLoaderLDS` / `RegTileLoader` split, mirroring `DenseALoader` /
# `Im2colALoader` in `matmul_kernel.mojo`). `Fp4WeightLoader` owns the packed-FP4
# weight -> dequant -> {register B fragment | SMEM strip} transition: it holds
# the `packed [N, K//2]` + `scales [N, ceil(K/16)]` TileTensor views and the K
# geometry, and does ALL addressing through TileTensor indexing / width-loads --
# no raw pointer arithmetic. The `_apple_frag_layout` register<->MMA-fragment
# transition stays at the SIMD layer in `matmul2d_mma_regc_bt_native` (KB
# `exceptions/apple-mma-fragment-is-not-distribute-expressible`: that bit-scatter
# is not `distribute`-expressible and is not pointer arithmetic to refactor away).
#
# The A activation is a plain row-major `(M, K)` tensor -- its per-lane fragment
# gather is a bounds-aware TileTensor indexed load owned here too (`load_a_frag`),
# which is where the ragged-M A over-read is fixed: OOB rows zero-fill (a zero A
# contributes nothing to the dot product, matching the dense reference's sum over
# valid rows), instead of the previous unguarded `a.ptr[gm*K+gk]`.


@fieldwise_init
struct Fp4WeightLoader[
    a_origin: ImmOrigin,
    packed_origin: ImmOrigin,
    scale_origin: ImmOrigin,
    //,
    in_type: DType,
    a_layout: TensorLayout,
    packed_layout: TensorLayout,
    scale_layout: TensorLayout,
    a_engine: TensorEngine,
    packed_engine: TensorEngine,
    scale_engine: TensorEngine,
](ImplicitlyCopyable, Movable):
    """Owner of the packed-FP4 weight -> dequant -> {register B | SMEM} transition.

    Holds the activation `a`, packed weight, and block-scale TileTensor views
    plus the `(M, N, K)` geometry, and exposes bounds-aware loads that do all
    addressing via TileTensor indexing (`t[i, j][0]`) and width-loads
    (`t.load[width=W](Coord(...))`) -- no raw pointer arithmetic. `in_type` is
    the dequant target (bf16). Each view's origin and engine are
    inferred struct parameters, so the kernel args are held exactly as passed.

    Parameters:
        a_origin: Origin of the activation view (inferred).
        packed_origin: Origin of the packed weight view (inferred).
        scale_origin: Origin of the block scales view (inferred).
        in_type: Dequant + activation dtype (bf16).
        a_layout: Layout of the activation `(M, K)` view.
        packed_layout: Layout of the packed weight `(N, K//2)` view.
        scale_layout: Layout of the block scales `(N, ceil(K/16))` view.
        a_engine: `TensorEngine` of the activation view.
        packed_engine: `TensorEngine` of the packed weight view.
        scale_engine: `TensorEngine` of the block scales view.
    """

    comptime SF = NVFP4_SF_VECTOR_SIZE  # 16

    # The origins are struct parameters because a field may not spell an
    # any-origin directly (same rule `Int8DequantWriter` parameterizes around).
    # The kernel args they bind to outlive the K-loop.
    var a: TileTensor[
        Self.in_type,
        Self.a_layout,
        Self.a_origin,
        Engine=Self.a_engine,
    ]
    var packed: TileTensor[
        .uint8,
        Self.packed_layout,
        Self.packed_origin,
        Engine=Self.packed_engine,
    ]
    var scales: TileTensor[
        .float8_e4m3fn,
        Self.scale_layout,
        Self.scale_origin,
        Engine=Self.scale_engine,
    ]
    var M: Int
    var N: Int
    var K: Int

    @always_inline
    @staticmethod
    def from_kernel_args(
        a: TileTensor[
            Self.in_type, Self.a_layout, Self.a_origin, Engine=Self.a_engine
        ],
        packed: TileTensor[
            .uint8,
            Self.packed_layout,
            Self.packed_origin,
            Engine=Self.packed_engine,
        ],
        scales: TileTensor[
            .float8_e4m3fn,
            Self.scale_layout,
            Self.scale_origin,
            Engine=Self.scale_engine,
        ],
        M: Int,
        N: Int,
        K: Int,
    ) -> Self:
        """Build the loader from the kernel's `AnyOrigin` tensor args.

        The fields carry the args' own origin and engine, so each view
        is held as passed -- no origin cast, and any engine carries
        through unchanged.

        Args:
            a: Bf16 activation `TileTensor` view with shape `(M, K)`.
            packed: Packed FP4 weight `TileTensor` view with shape `(N, K//2)`
                (uint8, lo-nibble first).
            scales: FP8-E4M3 block-scale `TileTensor` view with shape
                `(N, ceil(K/16))`.
            M: Number of rows in the activation and the output.
            N: Number of columns in the output and rows of the weight.
            K: Reduction dimension; inner size of `a` and the weight.
        """
        return Self(a, packed, scales, M, N, K)

    @always_inline
    def load_a_frag[
        bounded: Bool
    ](
        self,
        arow0: Int,
        k0: Int,
        a_rc: Array[IndexList[2], 8],
    ) -> SIMD[
        Self.in_type, 8
    ]:
        """This lane's 8-element A fragment for the 16x16 A tile at `(arow0, k0)`.

        `a_rc[i]` is the lane's `_apple_frag_layout` (row, col) for element `i`.
        On the interior (`bounded=False`) every row is in-bounds and the load is
        unconditional (branch-free). On an edge tile (`bounded=True`) rows past
        `M` zero-fill -- the ragged-M A over-read fix (a zero A element multiplies
        into nothing, matching the dense reference). K is always in-bounds
        (callers pass K % 16 == 0 tile-aligned strips).

        Parameters:
            bounded: Whether to bounds-check rows against `M` (edge tile).

        Args:
            arow0: Absolute M-origin (row) of the 16x16 A tile.
            k0: Absolute K-origin of the current K strip (a multiple of 16).
            a_rc: This lane's per-element `(row, col)` coords in the A tile
                (the `_apple_frag_layout` map, 8 entries).
        """
        var v = SIMD[Self.in_type, 8](0)
        comptime for i in range(8):
            var gm = arow0 + a_rc[i][0]
            var gk = k0 + a_rc[i][1]
            comptime if bounded:
                if gm < self.M:
                    v[i] = self.a[gm, gk][0].cast[Self.in_type]()
            else:
                v[i] = self.a[gm, gk][0].cast[Self.in_type]()
        return v

    @always_inline
    def decode_b_frag_regc(
        self,
        bcol0: Int,
        k0: Int,
        rb: Int,
        cb: Int,
    ) -> SIMD[Self.in_type, 16]:
        """Decode this lane's 16-element `transpose_right` register B fragment.

        The fragment reads 4 N-rows (`rb, rb+8, rb+16, rb+24`), each contributing
        4 CONTIGUOUS K values (`cb .. cb+3`, `cb` a multiple of 4). Those 4 K
        nibbles are one 2-byte packed load, and (because `cb+3 < 16`) share one
        FP8 block scale. All addressing is TileTensor width-loads / indexing.
        Interior fast path: callers pass tile-aligned N and K % 16 == 0, so the
        4 N-rows and the K run are always in-bounds.

        Args:
            bcol0: Absolute N-origin of the simdgroup's 16x32 B tile in the
                packed weight.
            k0: Absolute K-origin of the current K strip (a multiple of 16).
            rb: This lane's base N-row offset within the B tile
                (`frag_row_base(lane)`, range 0..7).
            cb: This lane's base K-column offset within the 16x16 sub-fragment
                (`frag_col_base(lane)`, a multiple of 4).
        """
        var v = SIMD[Self.in_type, 16](0)
        comptime for blk in range(4):
            comptime n_off = (blk % 2) * 8 + (blk // 2) * 16
            var n_abs = bcol0 + rb + n_off
            var k_abs0 = k0 + cb  # even (k0 mult 16, cb mult 4)
            # One 2-byte packed load -> 4 nibbles (k = cb .. cb+3).
            var two = self.packed.load[width=2, alignment=1](
                Coord(n_abs, k_abs0 // 2)
            )
            var pw = UInt16(two[0]) | (UInt16(two[1]) << UInt16(8))
            var nib = SIMD[.uint16, 4](0)
            nib[0] = pw & UInt16(0xF)  # k+0 (lo)
            nib[1] = (pw >> UInt16(4)) & UInt16(0xF)  # k+1 (hi)
            nib[2] = (pw >> UInt16(8)) & UInt16(0xF)  # k+2 (lo)
            nib[3] = (pw >> UInt16(12)) & UInt16(0xF)  # k+3 (hi)
            # One block scale for all 4 K (cb+3 < 16 -> same 16-block).
            var scale_abs = abs(
                self.scales[n_abs, k_abs0 // Self.SF][0].cast[.float32]()
            )
            var dec = (decode_e2m1_to_f32(nib) * scale_abs).cast[Self.in_type]()
            comptime for e in range(4):
                v[blk * 4 + e] = dec[e]
        return v

    @always_inline
    def decode_strip_to_smem[
        b_view_origin: Origin[mut=True],
        b_view_layout: TensorLayout,
        b_view_engine: TensorEngine,
        b_view_addr: AddressSpace,
        //,
        bytes_per_thread: Int,
        cols_per_thread: Int,
    ](
        self,
        b_view: TileTensor[
            Self.in_type,
            b_view_layout,
            b_view_origin,
            Engine=b_view_engine,
            address_space=b_view_addr,
        ],
        n_abs: Int,
        n_local: Int,
        k0: Int,
        col0: Int,
    ):
        """Cooperatively decode this thread's `cols_per_thread` bf16 cols of the
        `(BN, BK)` weight strip into the SMEM view `b_view`.

        This thread owns N-row `n_local` (absolute `n_abs`), a CONTIGUOUS
        `bytes_per_thread`-byte packed run at bf16 col `col0` (so adjacent threads
        read adjacent packed bytes -- the coalesced-K load), decoded with ONE FP8
        block scale (`cols_per_thread <= 16` = one 16-block). The packed run is a
        single TileTensor width-load; the SMEM write is a TileTensor width-store
        into `b_view` (no raw pointer arithmetic). All-in-bounds interior strip
        (callers pass tile-aligned N, K % BK == 0).

        Parameters:
            b_view_origin: Origin of the `b_view` SMEM store target.
            b_view_layout: Layout of the `b_view` SMEM store target.
            b_view_engine: `TensorEngine` of the `b_view` SMEM store target.
            b_view_addr: Address space of the `b_view` SMEM store target.
            bytes_per_thread: Packed bytes this thread loads in one width-load
                (each byte yields two bf16 columns).
            cols_per_thread: Number of bf16 columns this thread decodes and
                stores (`2 * bytes_per_thread`; must be <= 16 for one scale
                block).

        Args:
            b_view: SMEM `TileTensor` view of the `(BN, BK)` decoded weight
                strip to store into.
            n_abs: Absolute N-row index in the packed weight for this thread's
                strip (threadgroup N-origin + local row).
            n_local: Local N-row index within the `b_view` strip (the SMEM
                store coordinate).
            k0: Absolute K-origin of the current strip (a multiple of `BK`).
            col0: First bf16 column this thread decodes in its row's strip
                (a multiple of `COLS_PER_THREAD`).
        """
        var byte0 = col0 // 2  # first packed byte in this row's strip
        var bytes = self.packed.load[width=bytes_per_thread, alignment=1](
            Coord(n_abs, (k0 // 2) + byte0)
        )
        var nib = SIMD[.uint16, cols_per_thread](0)
        comptime for j in range(bytes_per_thread):
            var bj = UInt16(bytes[j])
            nib[2 * j] = bj & UInt16(0xF)
            nib[2 * j + 1] = (bj >> UInt16(4)) & UInt16(0xF)
        var scale_abs = abs(
            self.scales[n_abs, (k0 + col0) // Self.SF][0].cast[.float32]()
        )
        var dec = (decode_e2m1_to_f32(nib) * scale_abs).cast[Self.in_type]()
        b_view.store[width=cols_per_thread](Coord(n_local, col0), dec)


struct Matmul2dFp4[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    num_sg_m: Int = 2,
    num_sg_n: Int = 2,
    tm: Int = 2,
    tn: Int = 2,
    smem_bk: Int = 16,
]:
    """W4A16 GEMM on the native 16x32x16 `matmul2d` MMA: coalesced NT FP4 decode.

    Each 16x32x16 MMA is two native `_mma_apple_transposable`
    (`air.simdgroup_matrix_16x16x16`) calls (`matmul2d_mma_regc_bt_native`) --
    pure Mojo over `_mma_apple`, no external library or codegen dependency.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract). Currently unused on the interior store; wired for parity.
        num_sg_m: Simdgroup rows per threadgroup.
        num_sg_n: Simdgroup cols per threadgroup.
        tm: Output 16x32 tiles per simdgroup along M (register-C accumulators).
        tn: Output 16x32 tiles per simdgroup along N. `tm*tn == 4` is the M5
            register-cliff optimum; do NOT exceed 4 (measured collapse past it).
        smem_bk: Cooperative-decode strip depth for `run_smem_decode` (the bf16
            weight sub-tile columns decoded per phase; one decode feeds
            `smem_bk//16` MMA K-steps). Unused by the per-lane `run` kernel. The
            production launcher pins 256, the M5 optimum: a deeper strip
            amortizes the decode over more MMA work; 512 spills.
    """

    comptime in_type = DType.bfloat16  # dequant target + activation dtype
    comptime MMA_M = M2D_M  # 16
    comptime MMA_N = M2D_N  # 32
    comptime MMA_K = M2D_K  # 16
    comptime SF = NVFP4_SF_VECTOR_SIZE  # 16

    comptime NUM_SG = Self.num_sg_m * Self.num_sg_n
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE
    # Threadgroup output tile.
    comptime TG_M = Self.MMA_M * Self.num_sg_m * Self.tm
    comptime TG_N = Self.MMA_N * Self.num_sg_n * Self.tn

    # Cooperative-decode SMEM (run_smem_decode): the strip's decoded bf16 weight
    # sub-tile is (BN, BK) row-major, BN == TG_N. BK is the strip depth (cols
    # decoded per cooperative phase); one decode feeds BK//16 matmul2d K-steps.
    comptime BN = Self.TG_N
    # BK is the cooperative-decode strip depth (one decode feeds BK//16 matmul2d
    # K-steps -> deeper BK amortizes the barrier + un-overlappable decode over
    # more MMA work, the committed kernel's +34-45% BK=64 lever). Constraint:
    # COLS_PER_THREAD = BK // THREADS_PER_ROW = BK // (THREADS_PER_BLOCK // BN)
    # must be <= 16 (M5 16-bit SIMD width limit + one scale block per run). So a
    # deeper BK needs more threads per row: BK=32 -> >= 8 SG (256 threads at
    # BN=128), BK=64 -> >= 16 SG. Set via `num_sg_m` + `smem_bk` in enqueue_*_smem.
    comptime BK = Self.smem_bk
    # Cooperative decode work split: each N-row of the strip has BK//2 packed
    # bytes; split contiguously across THREADS_PER_ROW threads so ADJACENT
    # threads read ADJACENT packed bytes (coalesced K), unlike the per-lane
    # scattered-N decode in `run`.
    comptime PACKED_COLS = Self.BK // 2
    comptime THREADS_PER_ROW = Self.THREADS_PER_BLOCK // Self.BN
    comptime BYTES_PER_THREAD = Self.PACKED_COLS // Self.THREADS_PER_ROW
    comptime COLS_PER_THREAD = Self.BYTES_PER_THREAD * 2

    @__name(t"apple_matmul2d_fp4_run_{Self.c_type}")
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        packed_layout: TensorLayout,
        scale_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        packed_engine: TensorEngine,
        scale_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        packed: TileTensor[
            .uint8, packed_layout, ImmutAnyOrigin, Engine=packed_engine
        ],
        scales: TileTensor[
            .float8_e4m3fn, scale_layout, ImmutAnyOrigin, Engine=scale_engine
        ],
        M_arg: Int32,
        N_arg: Int32,
        K_arg: Int32,
    ):
        """W4A16 kernel entry. C `(M, N)`, A `(M, K)` bf16, packed `(N, K//2)`,
        scales `(N, ceil(K/16))`. Interior fast path (tile-aligned N, K%16==0).

        Parameters:
            c_layout: Layout of the output `C` `(M, N)` view.
            a_layout: Layout of the activation `A` `(M, K)` view.
            packed_layout: Layout of the packed FP4 weight `(N, K//2)` view.
            scale_layout: Layout of the block scales `(N, ceil(K/16))` view.
            c_engine: `TensorEngine` of the output `C` view.
            a_engine: `TensorEngine` of the activation `A` view.
            packed_engine: `TensorEngine` of the packed FP4 weight view.
            scale_engine: `TensorEngine` of the block scales view.

        Args:
            c: Output `TileTensor` view with shape `(M, N)`.
            a: Bf16 activation `TileTensor` view with shape `(M, K)`.
            packed: Packed FP4 weight `TileTensor` view with shape `(N, K//2)`
                (uint8, lo-nibble first).
            scales: FP8-E4M3 block-scale `TileTensor` view with shape
                `(N, ceil(K/16))`.
            M_arg: Number of rows in the activation and the output.
            N_arg: Number of columns in the output and rows of the weight.
            K_arg: Reduction dimension; inner size of `a` and the weight.
        """
        var M = Int(M_arg)
        var N = Int(N_arg)
        var K = Int(K_arg)
        var lane = Int(lane_id())
        var sg_id = Int(thread_idx.x) // WARP_SIZE
        var sg_m = sg_id // Self.num_sg_n
        var sg_n = sg_id % Self.num_sg_n

        var sg_row0 = (Int(block_idx.y) * Self.num_sg_m + sg_m) * (
            Self.MMA_M * Self.tm
        )
        var sg_col0 = (Int(block_idx.x) * Self.num_sg_n + sg_n) * (
            Self.MMA_N * Self.tn
        )

        # A frag: base map. B frag: transpose_right map (n, k) w/ k contiguous.
        var a_rc = Array[_, 8](
            fill_with_unrolled=lambda [i: Int]() -> IndexList[2]: a_frag_coord(
                lane, i
            )
        )
        # C store: UNCHANGED base bc map (transpose_right permutes only B).
        var c_rc = Array[_, 16](
            fill_with_unrolled=lambda [i: Int]() -> IndexList[2]: bc_frag_coord(
                lane, i
            )
        )

        # This lane's fragment (n, k) base for the transpose_right B map. The 16
        # B-frag elements are 4 N-rows x 4 contiguous K, one 2-byte packed load +
        # one scale per N-row. rb/cb are the base of the 16x16 sub-frag.
        var rb = frag_row_base(lane)  # 0..7
        var cb = frag_col_base(lane)  # {0,4,8,12}

        # The FP4 decode / A-gather owner: all packed/scale/A addressing goes
        # through it via TileTensor indexing (no raw pointer arithmetic).
        var loader = Fp4WeightLoader[
            Self.in_type,
            a_layout,
            packed_layout,
            scale_layout,
            a_engine,
            packed_engine,
            scale_engine,
        ].from_kernel_args(a, packed, scales, M, N, K)

        var accs = Array[_, Self.tm * Self.tn](fill=SIMD[.float32, 16](0))

        var k0 = 0
        while k0 < K:
            # A fragments (bf16 activation, base map). `bounded=True`: `run`
            # rounds the grid up, so a partial-M tile can gather rows past M --
            # zero-fill them (the ragged-M A over-read fix) rather than reading
            # OOB. (The reg `run` is the test-only path; the deep-K production
            # path is `run_smem_decode` below.)
            var a_frag = Array[_, Self.tm](
                fill_with_unrolled=lambda [im: Int]() -> SIMD[
                    .bfloat16, 8
                ]: loader.load_a_frag[bounded=True](
                    sg_row0 + im * Self.MMA_M, k0, a_rc
                )
            )

            # B fragments: decode packed FP4 -> bf16, K-contiguous (coalesced).
            var b_frag = Array[_, Self.tn](
                fill_with_unrolled=lambda [jn: Int]() -> SIMD[
                    .bfloat16, 16
                ]: loader.decode_b_frag_regc(
                    sg_col0 + jn * Self.MMA_N, k0, rb, cb
                )
            )

            comptime for im in range(Self.tm):
                comptime for jn in range(Self.tn):
                    comptime t = im * Self.tn + jn
                    matmul2d_mma_regc_bt_native(a_frag[im], b_frag[jn], accs[t])
            k0 += Self.MMA_K

        # Store (base bc C map). Guard < M / < N for the partial last tile.
        comptime for im in range(Self.tm):
            comptime for jn in range(Self.tn):
                comptime t = im * Self.tn + jn
                var trow0 = sg_row0 + im * Self.MMA_M
                var tcol0 = sg_col0 + jn * Self.MMA_N
                var cv = accs[t]
                comptime for i in range(16):
                    var gm = trow0 + c_rc[i][0]
                    var gn = tcol0 + c_rc[i][1]
                    if gm < M and gn < N:
                        c.store[width=1](
                            Coord(gm, gn),
                            SIMD[Self.c_type, 1](cv[i].cast[Self.c_type]()),
                        )

    @__name(t"apple_matmul2d_fp4_smemdec_{Self.c_type}")
    @staticmethod
    def run_smem_decode[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        packed_layout: TensorLayout,
        scale_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        packed_engine: TensorEngine,
        scale_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        packed: TileTensor[
            .uint8, packed_layout, ImmutAnyOrigin, Engine=packed_engine
        ],
        scales: TileTensor[
            .float8_e4m3fn, scale_layout, ImmutAnyOrigin, Engine=scale_engine
        ],
        M_arg: Int32,
        N_arg: Int32,
        K_arg: Int32,
    ):
        """Cooperative-decode W4A16: coalesced FP4 decode -> SMEM -> register B.

        The `run` kernel decodes per-lane, but each lane's `transpose_right` B
        fragment reads 4 N-scattered rows (stride-K/2 in the packed `(N, K//2)`
        weight) -> the packed + scale loads are uncoalesced gathers and dominate
        (measured: they drop the ~60 TF/s pure-MMA feed to ~10). This kernel
        COOPERATIVELY decodes the `(BN, BK)` weight sub-tile into SMEM with
        ADJACENT threads reading ADJACENT packed bytes (coalesced K), then each
        simdgroup reads its `tn` register B fragments from SMEM via the
        `transpose_right` map (`bt_frag_coord`) and runs the native 16x32x16 MMA
        (`matmul2d_mma_regc_bt_native`, two `_mma_apple_transposable` calls).
        This is the committed `AppleM5Fp4MatMul` cooperative-decode idea on the
        `matmul2d` tile shape -- the MMA stays REGISTER-fed (a plain
        SMEM->register indexed load).

        Interior fast path (tile-aligned N, K % BK == 0). A loads register-direct
        from DRAM (base map, K contiguous). Bit-exact vs `run` / the materialize
        oracle (same `decode_e2m1_to_f32 * |scale|`, same MMA), modulo the fp32
        MMA reduction order (identical to `run`, so bit-exact vs `run`).

        Parameters:
            c_layout: Layout of the output `C` `(M, N)` view.
            a_layout: Layout of the activation `A` `(M, K)` view.
            packed_layout: Layout of the packed FP4 weight `(N, K//2)` view.
            scale_layout: Layout of the block scales `(N, ceil(K/16))` view.
            c_engine: `TensorEngine` of the output `C` view.
            a_engine: `TensorEngine` of the activation `A` view.
            packed_engine: `TensorEngine` of the packed FP4 weight view.
            scale_engine: `TensorEngine` of the block scales view.

        Args:
            c: Output `TileTensor` view with shape `(M, N)`.
            a: Bf16 activation `TileTensor` view with shape `(M, K)`.
            packed: Packed FP4 weight `TileTensor` view with shape `(N, K//2)`
                (uint8, lo-nibble first).
            scales: FP8-E4M3 block-scale `TileTensor` view with shape
                `(N, ceil(K/16))`.
            M_arg: Number of rows in the activation and the output.
            N_arg: Number of columns in the output and rows of the weight.
            K_arg: Reduction dimension; inner size of `a` and the weight.
        """
        var M = Int(M_arg)
        var N = Int(N_arg)
        var K = Int(K_arg)
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SF = Self.SF
        comptime THREADS_PER_ROW = Self.THREADS_PER_ROW
        comptime BYTES_PER_THREAD = Self.BYTES_PER_THREAD
        comptime COLS_PER_THREAD = Self.COLS_PER_THREAD
        # M5 16-bit SIMD width limit + one scale block per thread's run.
        comptime assert COLS_PER_THREAD <= SF and (SF % COLS_PER_THREAD) == 0, (
            "cooperative decode needs COLS_PER_THREAD <= 16 (one scale block +"
            " M5 16-lane 16-bit SIMD limit)"
        )

        var lane = Int(lane_id())
        var tid = Int(thread_idx.x)
        var sg_id = tid // WARP_SIZE
        var sg_m = sg_id // Self.num_sg_n
        var sg_n = sg_id % Self.num_sg_n

        var tg_col = Int(block_idx.x) * BN  # N-origin of this threadgroup tile
        var sg_row0 = (Int(block_idx.y) * Self.num_sg_m + sg_m) * (
            Self.MMA_M * Self.tm
        )

        # SMEM: decoded bf16 (BN, BK) weight sub-tile, row-major. Wrapped in a
        # TileTensor view so the cooperative decode store and the per-SG B read
        # are TileTensor indexed (`b_view[n, col]`), in-bounds by construction --
        # no raw SMEM pointer arithmetic.
        var b_sm = unsafe_stack_allocation[
            BN * BK, Scalar[Self.in_type], address_space=.SHARED
        ]()
        var b_view = TileTensor(b_sm, Layout(Coord(BN, BK), Coord(BK, Idx[1])))

        var a_rc = Array[_, 8](
            fill_with_unrolled=lambda [i: Int]() -> IndexList[2]: a_frag_coord(
                lane, i
            )
        )
        # B fragment (n, k) local coords under transpose_right; k contiguous.
        var b_nk = Array[_, 16](
            fill_with_unrolled=lambda [i: Int]() -> IndexList[2]: bt_frag_coord(
                lane, i
            )
        )
        var c_rc = Array[_, 16](
            fill_with_unrolled=lambda [i: Int]() -> IndexList[2]: bc_frag_coord(
                lane, i
            )
        )

        # The FP4 decode / A-gather owner: all packed/scale/A/SMEM addressing
        # goes through it via TileTensor indexing (no raw pointer arithmetic).
        var loader = Fp4WeightLoader[
            Self.in_type,
            a_layout,
            packed_layout,
            scale_layout,
            a_engine,
            packed_engine,
            scale_engine,
        ].from_kernel_args(a, packed, scales, M, N, K)

        var accs = Array[_, Self.tm * Self.tn](fill=SIMD[.float32, 16](0))

        # This thread's cooperative-decode slot: N-row + contiguous byte-run.
        var dec_nrow = tid // THREADS_PER_ROW
        var dec_col_in_row = tid % THREADS_PER_ROW
        var dec_col0 = dec_col_in_row * COLS_PER_THREAD  # first bf16 col
        var dec_n_abs = tg_col + dec_nrow

        # Whether THIS threadgroup's M-tile runs past M (ragged-M edge). Decided
        # ONCE (not per-load) so the interior K-loop A-gather is branch-free: the
        # `bounded=False` instantiation contains no bounds branch (comptime),
        # matching the AppleM5Fp4MatMul edge/interior split. The B decode/read is
        # always interior (tile-aligned N + K % BK == 0).
        var tg_m_end = (Int(block_idx.y) + 1) * Self.TG_M
        var is_m_edge = tg_m_end > M

        @always_inline
        @__parameter
        def _kloop[bounded: Bool]():
            var k0 = 0
            while k0 < K:
                # ---- cooperative coalesced decode of (BN, BK) -> b_view ----
                loader.decode_strip_to_smem[BYTES_PER_THREAD, COLS_PER_THREAD](
                    b_view, dec_n_abs, dec_nrow, k0, dec_col0
                )
                barrier()

                # ---- per-SG: read tn register B frags from SMEM, matmul2d ----
                var ks = 0
                while ks < BK:
                    var a_frag = Array[_, Self.tm](
                        fill_with_unrolled=lambda [im: Int]() -> SIMD[
                            Self.in_type, 8
                        ]: loader.load_a_frag[bounded=bounded](
                            sg_row0 + im * Self.MMA_M, k0 + ks, a_rc
                        )
                    )

                    def b_frag_at[jn: Int]() {imm} -> SIMD[Self.in_type, 16]:
                        # SG's N-subtile within the (BN) strip.
                        var n_local0 = (sg_n * Self.tn + jn) * Self.MMA_N
                        var v = SIMD[Self.in_type, 16](0)
                        comptime for i in range(16):
                            var nl = n_local0 + b_nk[i][0]  # local N in [0, BN)
                            var kl = ks + b_nk[i][1]  # local K in [0, BK)
                            v[i] = b_view[nl, kl][0]
                        return v

                    var b_frag = Array[_, Self.tn](fill_with_unrolled=b_frag_at)

                    comptime for im in range(Self.tm):
                        comptime for jn in range(Self.tn):
                            comptime t = im * Self.tn + jn
                            matmul2d_mma_regc_bt_native(
                                a_frag[im], b_frag[jn], accs[t]
                            )
                    ks += Self.MMA_K
                barrier()
                k0 += BK

        if is_m_edge:
            _kloop[True]()
        else:
            _kloop[False]()

        var sg_col0 = tg_col + sg_n * (Self.MMA_N * Self.tn)
        comptime for im in range(Self.tm):
            comptime for jn in range(Self.tn):
                comptime t = im * Self.tn + jn
                var trow0 = sg_row0 + im * Self.MMA_M
                var tcol0 = sg_col0 + jn * Self.MMA_N
                var cv = accs[t]
                comptime for i in range(16):
                    var gm = trow0 + c_rc[i][0]
                    var gn = tcol0 + c_rc[i][1]
                    if gm < M and gn < N:
                        c.store[width=1](
                            Coord(gm, gn),
                            SIMD[Self.c_type, 1](cv[i].cast[Self.c_type]()),
                        )


@always_inline
def enqueue_matmul2d_fp4[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    num_sg_m: Int = 2,
    num_sg_n: Int = 2,
    tm: Int = 2,
    tn: Int = 2,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the `matmul2d` W4A16 GEMM: `out = a @ dequant(packed, scales)^T`.

    `a` is the bf16 activation `(M, K)`, `packed` the FP4 weight `(N, K//2)`
    (lo-nibble first), `scales` the FP8-E4M3 block scales `(N, ceil(K/16))`. C is
    `(M, N)`. Interior-only fast path: caller must ensure N is a multiple of the
    threadgroup N-tile (`32*num_sg_n*tn`) and K % 16 == 0 (the W4A16 production
    shapes are aligned). M is rounded up and the store guards partial tiles.

    The 16x32x16 MMA is the native `matmul2d_mma_regc_bt_native` tiling (two
    `_mma_apple_transposable` calls); pure Mojo over `_mma_apple`.

    Parameters:
        c_type: Output element type; one of `fp16`, `bf16`, `fp32` (defaults
            to `float32`). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue applying a per-element
            transform to the output (defaults to `None`).
        num_sg_m: Simdgroup rows per threadgroup (defaults to 2).
        num_sg_n: Simdgroup cols per threadgroup (defaults to 2).
        tm: Output 16x32 tiles per simdgroup along M (defaults to 2).
        tn: Output 16x32 tiles per simdgroup along N (defaults to 2).
            `tm*tn == 4` is the M5 register-cliff optimum.

    Args:
        c: Output `TileTensor` view with shape `(M, N)`.
        a: Bf16 activation `TileTensor` view with shape `(M, K)`.
        packed: Packed FP4 weight `TileTensor` view with shape `(N, K//2)`
            (uint8, lo-nibble first).
        scales: FP8-E4M3 block-scale `TileTensor` view with shape
            `(N, ceil(K/16))`.
        ctx: Device context used to enqueue the kernel.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime MM = Matmul2dFp4[
        c_type,
        elementwise_lambda_fn,
        num_sg_m,
        num_sg_n,
        tm,
        tn,
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_matmul2d_fp4 requires Apple M5 (compute_capability =="
                " 5); got compute_capability="
            ),
            cc,
        )

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_matmul2d_fp4: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(packed.dim[0]()) == n, "packed must be (N, K//2)")
    debug_assert(Int(packed.dim[1]()) == k // 2, "packed must be (N, K//2)")
    debug_assert(Int(scales.dim[0]()) == n, "scales must be (N, ceil(K/16))")

    var grid_m = ceildiv(m, MM.TG_M)
    var grid_n = ceildiv(n, MM.TG_N)

    comptime kernel = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(packed).LayoutType,
        type_of(scales).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(packed).Engine,
        type_of(scales).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        packed.as_immut(),
        scales.as_immut(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(grid_n, grid_m),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


@always_inline
def enqueue_matmul2d_fp4_smem[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    # Production default = the M5-Max-tuned geometry: 16 simdgroups (512 threads)
    # sharing a NARROW BN=32 decoded B tile (tm=4, tn=1) with a DEEP smem_bk=256
    # strip. Amortizes the barrier + un-overlappable decode over 16 co-resident
    # M-simdgroups + smem_bk//16=16 MMA K-steps per decode.
    #
    # smem_bk=256 is the M5 optimum: COLS_PER_THREAD = smem_bk / (NUM_SG*32/BN)
    # = 256/(512/32) = 16, the M5 16-lane 16-bit-SIMD-width MAX at 512 threads. A
    # deeper strip amortizes the decode over more MMA work (halves the barrier +
    # decode-phase count per unit of MMA work), bit-exact vs the materialize->
    # dense oracle. Deeper still (BK=512) needs 1024 threads (to keep COLS <=16)
    # and regresses on M5 occupancy; wider N (tn>1) blows the M5 4-accumulator
    # register cliff. So 16SG/BN32/tm4/tn1/BK256 is the pinned optimum. Requires
    # K % 256 == 0; all W4A16 production shapes satisfy it (FLUX K in
    # {3072,6144,12288,18432} are all multiples of 256).
    num_sg_m: Int = 16,
    num_sg_n: Int = 1,
    tm: Int = 4,
    tn: Int = 1,
    smem_bk: Int = 256,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the cooperative-decode `matmul2d` W4A16 GEMM (`run_smem_decode`).

    Coalesced FP4 decode -> SMEM -> register B fragment (the fast W4A16 path;
    the per-lane `run` is decode-load-bound). Interior-only: caller must ensure
    N is a multiple of the threadgroup N-tile (`32*num_sg_n*tn`) and K % smem_bk
    == 0 (smem_bk is the cooperative-decode strip depth).

    The 16x32x16 MMA is the native `matmul2d_mma_regc_bt_native` tiling (two
    `_mma_apple_transposable` calls); pure Mojo over `_mma_apple`.

    Parameters:
        c_type: Output element type; one of `fp16`, `bf16`, `fp32` (defaults
            to `float32`). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue applying a per-element
            transform to the output (defaults to `None`).
        num_sg_m: Simdgroup rows per threadgroup (defaults to 16).
        num_sg_n: Simdgroup cols per threadgroup (defaults to 1).
        tm: Output 16x32 tiles per simdgroup along M (defaults to 4).
        tn: Output 16x32 tiles per simdgroup along N (defaults to 1).
        smem_bk: Cooperative-decode strip depth in K; one decode feeds
            `smem_bk//16` MMA K-steps (defaults to 256).

    Args:
        c: Output `TileTensor` view with shape `(M, N)`.
        a: Bf16 activation `TileTensor` view with shape `(M, K)`.
        packed: Packed FP4 weight `TileTensor` view with shape `(N, K//2)`
            (uint8, lo-nibble first).
        scales: FP8-E4M3 block-scale `TileTensor` view with shape
            `(N, ceil(K/16))`.
        ctx: Device context used to enqueue the kernel.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime MM = Matmul2dFp4[
        c_type,
        elementwise_lambda_fn,
        num_sg_m,
        num_sg_n,
        tm,
        tn,
        smem_bk,
    ]

    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "enqueue_matmul2d_fp4_smem requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
        )

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_matmul2d_fp4_smem: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(packed.dim[0]()) == n, "packed must be (N, K//2)")
    debug_assert(Int(packed.dim[1]()) == k // 2, "packed must be (N, K//2)")
    debug_assert(Int(scales.dim[0]()) == n, "scales must be (N, ceil(K/16))")

    var grid_m = ceildiv(m, MM.TG_M)
    var grid_n = ceildiv(n, MM.TG_N)

    comptime kernel = MM.run_smem_decode[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(packed).LayoutType,
        type_of(scales).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(packed).Engine,
        type_of(scales).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        packed.as_immut(),
        scales.as_immut(),
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(grid_n, grid_m),
        block_dim=(MM.THREADS_PER_BLOCK),
    )
