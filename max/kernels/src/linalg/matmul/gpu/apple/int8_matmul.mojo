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
"""Apple M5 int8 W8A8 matmul: `out = dequant(A_int8 @ B_int8^T)`.

Apple silicon GPU (Metal 4, `compute_capability == 5`). The int32-accumulate
sibling of the W4A16 kernel (`matmul2d_fp4.mojo` / `fp4_matmul.mojo`): where the
FP4 path decodes a packed 4-bit weight to bf16 and feeds the *float* simdgroup
MMA, this path feeds int8 A and int8 B straight to the M5 **integer widening**
simdgroup MMA (`_mma_apple` with int8 inputs -> int32 accumulator, emitting
`air.simdgroup_matrix_16x16x16_widening_multiply_accumulate`). That widening op
is a genuinely faster M5 datapath than the float MMA (~1.7-1.9x at the FLUX.2
projection/MLP shapes; the dequant epilogue nets ~1.5-1.7x). The dense
`AppleM5MatMul` cannot be reused: it hardcodes a *float* accumulator
(`MmaOpApple[float32, in_type]`) and asserts a float `c_type`.

Quantization (W8A8): both operands are symmetric-absmax int8 (`scale =
absmax / 127`), A per-row (per-token), B per-column (per-output-channel). The
kernel accumulates the raw int8xint8 products in int32, then the epilogue
DEQUANTIZES each output element by `a_scale[row] * b_scale[col]` and casts to
`c_type` (bf16), with an optional per-column bias added after dequant. A is
quantized online by `enqueue_apple_int8_quantize_activation` (a separate one-pass
kernel, reusing the `_quantize_a_block`-style absmax); B is pre-quantized at
weight load.

Structure mirrors `AppleM5MatMul._run_gemm_body`: 64x64 threadgroup block, 4
simdgroups (128 threads), each a 32x32 (2x2 `MmaOpApple`) subtile;
rectangular-Morton tile schedule (wide-N shapes are Morton-critical -- a flat
grid measured 50-70 Tops/s vs 96-99 with Morton); DRAM->register loads (no SMEM
staging -- the M5 rule). The K-loop feeds the widening MMA per `BK`=64 strip:
the `K % 16 == 0` interior takes a width-16 int8 K-repartition; the K-tail and
`K % 16 != 0` strips take an AGX3 edge-masked width-4 path; genuinely ragged M/N
tiles take the bounded path. See KB `kernels/apple-m5-int8-matmul`.
"""

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.compute.arch.mma_apple import _mma_apple_transposable
from max.gpu.host import DeviceContext
from max.gpu.memory import build_edge_mask, gmem_edge_masked_load
from std.collections import Optional
from std.math import ceildiv, round
from std.memory import unsafe_stack_allocation
from std.utils import IndexList

from layout import TensorEngine, TileTensor
from layout.coord import Coord
from layout.tile_layout import TensorLayout, row_major

from linalg.arch.apple.mma import MmaOpApple


def _require_apple_m5(ctx: DeviceContext) raises:
    """Runtime guard: reject a non-Apple-M5 GPU at the host enqueue entry.

    The kernel body uses the Apple simdgroup MMA (`_mma_apple`), which compiles
    for any Apple GPU target, so the Metal-4 / M5 requirement is enforced at
    launch (matching `enqueue_apple_matmul` / `enqueue_apple_fp4_matmul`), not
    with a comptime assert that would break a non-M5 Apple build.
    """
    var cc = ctx.compute_capability()
    if cc != 5:
        raise Error(
            (
                "Apple int8 W8A8 matmul requires Apple M5"
                " (compute_capability == 5); got compute_capability="
            ),
            cc,
        )


@fieldwise_init
struct Int8DequantWriter[
    c_origin: MutOrigin,
    s_origin: ImmOrigin,
    //,
    c_type: DType,
    c_layout: TensorLayout,
    as_layout: TensorLayout,
    bs_layout: TensorLayout,
    bias_layout: TensorLayout,
    c_engine: TensorEngine,
    as_engine: TensorEngine,
    bs_engine: TensorEngine,
    bias_engine: TensorEngine,
    *,
    has_bias: Bool,
](ImplicitlyCopyable, Movable):
    """Owns the C-output write of the W8A8 epilogue (the write-side counterpart
    to the FP4 `Fp4WeightLoader`).

    Encapsulates the int32-accumulator -> dequant -> `c_type` store: given a
    16x16 accumulator fragment and its tile origin, it applies the per-row
    activation scale x per-column weight scale (+ optional per-column bias) and
    stores each in-bounds element through the C `TileTensor`. All addressing is
    TileTensor indexing / `store` -- no raw pointer arithmetic and no origin
    rebase. The struct is generic over the view origins (`c_origin` mutable for
    the C write, `s_origin` immutable and shared by the scale/bias reads),
    inferred by `@fieldwise_init` from the constructor args, so the fields hold
    the caller's real origins directly (a plain fieldwise constructor, no
    origin-erasing helper needed). Per the KB TileIO pattern
    (`new-primitives/amd-tile-io-expert-objects`), every register->DRAM
    transition has an owner; this is that owner for the int8 GEMM.

    The `_apple_frag_layout` maps a lane's 8-element fragment to rows
    `{rb, rb+8}` x cols `{cb, cb+1, cb+2, cb+3}`: `frag[0:4]` = row `rb`,
    `frag[4:8]` = row `rb+8` (matching `MmaOpApple._store_fragment`).

    Parameters:
        c_origin: Mutable `MutOrigin` of the C output write view (inferred).
        s_origin: Immutable `ImmOrigin` shared by the scale and bias read
            views (inferred).
        c_type: Output element type (`bf16` / `fp16` / `fp32`).
        c_layout: `TensorLayout` of the C output `TileTensor`.
        as_layout: `TensorLayout` of the per-row activation scale `TileTensor`.
        bs_layout: `TensorLayout` of the per-column weight scale `TileTensor`.
        bias_layout: `TensorLayout` of the per-column bias `TileTensor`.
        c_engine: `TensorEngine` of the C output write view.
        as_engine: `TensorEngine` of the per-row activation scale view.
        bs_engine: `TensorEngine` of the per-column weight scale view.
        bias_engine: `TensorEngine` of the per-column bias view.
        has_bias: If `True`, add `bias[col]` to each output element after
            dequant.
    """

    var c: TileTensor[
        Self.c_type, Self.c_layout, Self.c_origin, Engine=Self.c_engine
    ]
    var a_scale: TileTensor[
        .float32,
        Self.as_layout,
        Self.s_origin,
        Engine=Self.as_engine,
    ]
    var b_scale: TileTensor[
        .float32,
        Self.bs_layout,
        Self.s_origin,
        Engine=Self.bs_engine,
    ]
    var bias: TileTensor[
        Self.c_type, Self.bias_layout, Self.s_origin, Engine=Self.bias_engine
    ]
    var M: Int
    var N: Int

    @always_inline
    def write_frag(
        self,
        frag: SIMD[.int32, 8],
        tile_r0: Int,
        tile_c0: Int,
        rb: Int,
        cb: Int,
    ):
        """Dequant + store one 16x16 accumulator fragment (this lane's 8 elems).

        Bounds-guards rows `< M` and cols `< N` for the partial last tile;
        width-4 store when the 4 cols are in-bounds, scalar on the N-edge.

        Args:
            frag: This lane's 8 int32 accumulator elements from the 16x16
                MMA fragment (`frag[0:4]` = row `rb`, `frag[4:8]` = row
                `rb+8`).
            tile_r0: Absolute row origin of the 16x16 accumulator tile in
                the C output.
            tile_c0: Absolute column origin of the 16x16 accumulator tile
                in the C output.
            rb: In-fragment row base offset for this lane; selects rows
                `rb` and `rb+8` within the tile.
            cb: In-fragment column base offset for this lane; selects the
                4-column group `{cb, cb+1, cb+2, cb+3}` within the tile.
        """
        comptime for half in range(2):
            var r = tile_r0 + rb + half * 8
            if r < self.M:
                var asc = self.a_scale.load[width=1](Coord(r))
                var col0 = tile_c0 + cb
                var fv = frag.slice[4, offset=half * 4]().cast[.float32]()
                if col0 + 3 < self.N:
                    var bs = self.b_scale.raw_load[width=4](col0)
                    var y4 = (fv * asc * bs).cast[Self.c_type]()
                    comptime if Self.has_bias:
                        y4 = y4 + self.bias.raw_load[width=4](col0)
                    self.c.store[width=4](Coord(r, col0), y4)
                else:
                    comptime for cc in range(4):
                        var col = col0 + cc
                        if col < self.N:
                            var v = (
                                Float32(fv[cc])
                                * asc
                                * self.b_scale.load[width=1](Coord(col))
                            )
                            var y = v.cast[Self.c_type]()
                            comptime if Self.has_bias:
                                y = y + self.bias.load[width=1](Coord(col))
                            self.c.store[width=1](
                                Coord(r, col), SIMD[Self.c_type, 1](y)
                            )

    @always_inline
    def write_frag_full(
        self,
        frag: SIMD[.int32, 8],
        tile_r0: Int,
        tile_c0: Int,
        rb: Int,
        cb: Int,
    ):
        """Interior full-tile store: dequant + width-4 store, no bounds check.

        The `run` caller invokes this only for `not is_edge` tiles, where every
        row and 4-col group is provably in-bounds -- so the per-fragment guards
        of `write_frag` are dropped. Bit-identical to `write_frag` here.
        """
        comptime for half in range(2):
            var r = tile_r0 + rb + half * 8
            var asc = self.a_scale.load[width=1](Coord(r))
            var col0 = tile_c0 + cb
            var fv = frag.slice[4, offset=half * 4]().cast[.float32]()
            var bs = self.b_scale.raw_load[width=4](col0)
            var y4 = (fv * asc * bs).cast[Self.c_type]()
            comptime if Self.has_bias:
                y4 = y4 + self.bias.raw_load[width=4](col0)
            self.c.store[width=4](Coord(r, col0), y4)


struct AppleM5Int8MatMul[
    c_type: DType = .bfloat16,
    *,
    has_bias: Bool = False,
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 64,
    TTI32: Bool = False,
]:
    """W8A8 GEMM: int8 A x int8 B^T -> int32 accum -> per-row/col dequant.

    Parameters:
        c_type: Output element type (bf16 / fp16 / fp32). Accumulation is int32;
            the dequant multiply is done in fp32 then cast to `c_type`.
        has_bias: If True, add a per-output-column bias (in `c_type`) after
            dequant.
        BM: Threadgroup M-tile height (multiple of `SG_M`).
        BN: Threadgroup N-tile width (multiple of `SG_N`).
        BK: K-strip depth per accumulate step (multiple of `MMA_K` = 16;
            default 64 = four 16-wide K-blocks).
        TTI32: If True, use int32 `load_linear` for the interior A/B loads (faster
            on NA-bound shapes, numerically identical). Auto-selected by
            `enqueue_apple_int8_matmul` when `max(M*K, N*K) < 2^31`, else i64.

    `run` is the GPU kernel entry (TileTensor operands + `M`/`N`/`K`). Launch via
    `enqueue_apple_int8_matmul`.
    """

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 16
    comptime SG_M = 32  # simdgroup subtile rows (2 * MMA_M)
    comptime SG_N = 32  # simdgroup subtile cols (2 * MMA_N)
    comptime NUM_MMA_M = Self.SG_M // Self.MMA_M  # 2
    comptime NUM_MMA_N = Self.SG_N // Self.MMA_N  # 2
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE

    # int8 -> int32 widening MMA (2x2 accumulators, the M5 register optimum).
    # NT only (`transpose_b=True`): the width-16 K-repartition needs contiguous K.
    comptime Mma = MmaOpApple[
        DType.int32,
        DType.int8,
        num_m_mmas=Self.NUM_MMA_M,
        num_n_mmas=Self.NUM_MMA_N,
        transpose_b=True,
    ]

    # === Morton (Z-order) tile scheduling (copied from `AppleM5MatMul`; wide-N
    # locality lever) ====================================================== #

    @staticmethod
    def morton_decode_2d(flat_idx: UInt32) -> Tuple[UInt32, UInt32]:
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
        flat_idx: UInt32, log2_m: UInt32, log2_n: UInt32
    ) -> Tuple[UInt32, UInt32]:
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

    # === Width-16 int8 K-repartition (interior fast path) ================== #
    # Interior `K % 16 == 0` strips only. Bit-identical to a 4x width-4 split;
    # why, and the align-16 gate, are in KB `kernels/apple-m5-int8-matmul`.

    @staticmethod
    @always_inline
    def _load_frag_x4_int8(
        strip: TileTensor[.int8, ...],
        row_stride: Int32,
        base_row: Int32,
        rb: Int32,
        four_cb: Int32,
    ) -> Array[SIMD[.int8, 8], 4]:
        """Width-16 load of one 16-block's four K-block fragments from a 64-wide
        K strip. `row_stride`/`four_cb` (= `4*cb`) are hoisted by the caller so
        the per-block offset is one mul-add; the offset math is Int32 (offset from
        the i64 tile-origin ptr, bounded << 2^31 by the k,n <= 65535 contract).
        """
        var lo_off = (base_row + rb) * row_stride + four_cb
        var hi_off = (base_row + rb + 8) * row_stride + four_cb
        # align=16 required: at align=1 the AGX JIT scalarizes the 16-lane load
        # into 16 byte-loads. Safe only because this path is gated on `k % 16 == 0`
        # (a misaligned address under align=16 is rounded down) -- see KB
        # kernels/apple-m5-int8-matmul.
        comptime align = 16
        var lo16 = strip.raw_load[width=16, alignment=align](Int(lo_off))
        var hi16 = strip.raw_load[width=16, alignment=align](Int(hi_off))
        var out = Array[SIMD[.int8, 8], 4](uninitialized=True)
        comptime for j in range(4):
            out[j] = lo16.slice[4, offset=4 * j]().join(
                hi16.slice[4, offset=4 * j]()
            )
        return out^

    @staticmethod
    @always_inline
    def _mma_width16(
        mut accum: Self.Mma.AccumType,
        a_strip: TileTensor[.int8, ...],
        b_strip: TileTensor[.int8, ...],
        rb: Int,
        cb: Int,
    ):
        """Interior BK=64 MMA via the width-16 K-repartition. Pre-loads all A/B
        K-block fragments, then issues the MMAs K-block-outer so consecutive MMAs
        write different accumulators (hides the dependent int32 accumulate behind
        independent MMAs). Reorder is bit-identical. hw_transpose_a=False (A (M,K)),
        hw_transpose_b=True (B (N,K), transpose_b).
        """
        # Hoist stride + K col-offset once, narrowed to Int32 (see loader).
        var a_rs = Int32(Self.Mma._row_stride(a_strip))
        var b_rs = Int32(Self.Mma._row_stride(b_strip))
        var rb32 = Int32(rb)
        var four_cb = Int32(4 * cb)

        var a_all = Array[SIMD[.int8, 8], Self.NUM_MMA_M * 4](
            uninitialized=True
        )
        var b_all = Array[SIMD[.int8, 8], Self.NUM_MMA_N * 4](
            uninitialized=True
        )
        comptime for mi in range(Self.NUM_MMA_M):
            var af = Self._load_frag_x4_int8(
                a_strip, a_rs, Int32(mi * Self.MMA_M), rb32, four_cb
            )
            comptime for j in range(4):
                a_all[mi * 4 + j] = af[j]
        comptime for ni in range(Self.NUM_MMA_N):
            var bf = Self._load_frag_x4_int8(
                b_strip, b_rs, Int32(ni * Self.MMA_N), rb32, four_cb
            )
            comptime for j in range(4):
                b_all[ni * 4 + j] = bf[j]
        comptime for j in range(4):
            comptime for ni in range(Self.NUM_MMA_N):
                comptime for mi in range(Self.NUM_MMA_M):
                    comptime idx = mi * Self.NUM_MMA_N + ni
                    _mma_apple_transposable(
                        accum[idx],
                        a_all[mi * 4 + j],
                        b_all[ni * 4 + j],
                        accum[idx],
                        False,
                        True,
                    )

    # === TTI32: int32-indexed interior load ================================ #
    # Default global-tensor TileTensor indexing lowers to i64; folding the
    # absolute A/B offset in int32 is the addressing win. See the `TTI32` param.
    @staticmethod
    @always_inline
    def _load_frag_x4_int8_abs(
        t32: TileTensor[.int8, ...],
        abs_row: Int,
        abs_k: Int,
    ) -> Array[SIMD[.int8, 8], 4]:
        """Int32-indexed `_load_frag_x4_int8`: `<16 x i8>` load via absolute
        `load_linear`. `align=16` keeps the vector load (needs `abs_k % 16 == 0`).
        """
        comptime align = 16
        var lo16 = t32.load_linear[width=16, alignment=align](
            IndexList[2](abs_row, abs_k)
        )
        var hi16 = t32.load_linear[width=16, alignment=align](
            IndexList[2](abs_row + 8, abs_k)
        )
        var out = Array[SIMD[.int8, 8], 4](uninitialized=True)
        comptime for j in range(4):
            out[j] = lo16.slice[4, offset=4 * j]().join(
                hi16.slice[4, offset=4 * j]()
            )
        return out^

    @staticmethod
    @always_inline
    def _mma_width16_abs(
        mut accum: Self.Mma.AccumType,
        a32: TileTensor[.int8, ...],
        b32: TileTensor[.int8, ...],
        sg_row: Int,
        sg_col: Int,
        ks: Int,
        rb: Int,
        cb: Int,
    ):
        """Int32-indexed `_mma_width16` (same preload + K-block-outer ILP order,
        so bit-identical); only the address form/width differ."""
        var abs_k = ks * Self.BK + 4 * cb
        var a_base = sg_row * Self.SG_M + rb
        var b_base = sg_col * Self.SG_N + rb
        var a_all = Array[SIMD[.int8, 8], Self.NUM_MMA_M * 4](
            uninitialized=True
        )
        var b_all = Array[SIMD[.int8, 8], Self.NUM_MMA_N * 4](
            uninitialized=True
        )
        comptime for mi in range(Self.NUM_MMA_M):
            var af = Self._load_frag_x4_int8_abs(
                a32, a_base + mi * Self.MMA_M, abs_k
            )
            comptime for j in range(4):
                a_all[mi * 4 + j] = af[j]
        comptime for ni in range(Self.NUM_MMA_N):
            var bf = Self._load_frag_x4_int8_abs(
                b32, b_base + ni * Self.MMA_N, abs_k
            )
            comptime for j in range(4):
                b_all[ni * 4 + j] = bf[j]
        comptime for j in range(4):
            comptime for ni in range(Self.NUM_MMA_N):
                comptime for mi in range(Self.NUM_MMA_M):
                    comptime idx = mi * Self.NUM_MMA_N + ni
                    _mma_apple_transposable(
                        accum[idx],
                        a_all[mi * 4 + j],
                        b_all[ni * 4 + j],
                        accum[idx],
                        False,
                        True,
                    )

    # === Edge-masked width-4 K-tail path (PR #91003, Apple M5 only) ========= #
    # Alignment-robust AGX3 edge-masked width-4 loads along contiguous K, for the
    # K-tail and `K % 16 != 0` strips (where the align-16 width-16 load is unsafe).
    # An edge tool, not the interior: masking every strip lost ~22%. See KB
    # kernels/apple-m5-int8-matmul.
    @staticmethod
    @always_inline
    def _masked_frag(
        sub: TileTensor[.int8, ...],
        row_stride: Int,
        rb: Int,
        cb: Int,
        valid_along_row: Int,
        kb_valid: Int,
    ) -> SIMD[.int8, 8]:
        """One 16x16 fragment via two width-4 edge-masked loads (row-halves `rb`,
        `rb+8`). The K mask zeroes K-lanes past `kb_valid`; row/col validity folds
        in by forcing mask=0 when the row-half is OOB along the non-K axis (M for
        A, N for B), so the helper serves both interior and edge tiles."""
        var kmask = build_edge_mask(Int32(cb), Int32(0), Int32(kb_valid))
        var lo_mask = kmask if rb < valid_along_row else Int16(0)
        var hi_mask = kmask if (rb + 8) < valid_along_row else Int16(0)
        var lo = gmem_edge_masked_load[4](
            sub.ptr + (rb * row_stride + cb), lo_mask
        )
        var hi = gmem_edge_masked_load[4](
            sub.ptr + ((rb + 8) * row_stride + cb), hi_mask
        )
        return lo.join(hi)

    @staticmethod
    @always_inline
    def _mma_masked(
        mut accum: Self.Mma.AccumType,
        a_strip: TileTensor[.int8, ...],
        b_strip: TileTensor[.int8, ...],
        rb: Int,
        cb: Int,
        valid_rows: Int,
        valid_cols: Int,
        k_valid: Int,
    ):
        """One BK strip via masked width-4 loads (`hw_transpose_a=False`,
        `hw_transpose_b=True`; K contiguous). Each `BK // MMA_K` K-block masks off
        lanes past `k_valid`, so a partial final K-block is zero-filled;
        bit-identical to the width-16 / bounded paths on the shared K range."""
        var a_rs = Self.Mma._row_stride(a_strip)
        var b_rs = Self.Mma._row_stride(b_strip)
        comptime num_kb = Self.BK // Self.MMA_K
        comptime for ki in range(num_kb):
            # Clamp to >= 0: `build_edge_mask(cb, 0, upper)` with upper < 0 returns
            # all-TRUE (AGX3 bound-compare footgun); upper == 0 is all-zero. See
            # KB kernels/apple-m5-int8-matmul.
            var kb_valid = max(
                0, min(Int(Self.MMA_K), k_valid - ki * Self.MMA_K)
            )
            var b_frags = Array[_, Self.NUM_MMA_N](
                fill_with_unrolled=lambda [ni: Int]() -> SIMD[
                    .int8, 8
                ]: Self._masked_frag(
                    b_strip.tile[16, 16](ni, ki),
                    b_rs,
                    rb,
                    cb,
                    valid_cols - ni * 16,
                    kb_valid,
                )
            )
            comptime for mi in range(Self.NUM_MMA_M):
                var a_sub = a_strip.tile[16, 16](mi, ki)
                var a_frag = Self._masked_frag(
                    a_sub, a_rs, rb, cb, valid_rows - mi * 16, kb_valid
                )
                comptime for ni in range(Self.NUM_MMA_N):
                    _mma_apple_transposable(
                        accum[mi * Self.NUM_MMA_N + ni],
                        a_frag,
                        b_frags[ni],
                        accum[mi * Self.NUM_MMA_N + ni],
                        False,
                        True,
                    )

    @staticmethod
    @always_inline
    def _accumulate[
        bounded: Bool
    ](
        mut accum: Self.Mma.AccumType,
        a: TileTensor[.int8, ...],
        b: TileTensor[.int8, ...],
        mma_op: Self.Mma,
        sg_row: Int,
        sg_col: Int,
        valid_rows: Int,
        valid_cols: Int,
        n_full_strips: Int,
        has_k_tail: Bool,
        k: Int,
        k16: Bool,
    ):
        """K-loop accumulate over the SG_M x SG_N subtile. `accum` is passed `mut`
        by argument, NOT captured by a closure: a closure capturing `accum` pins
        it to the stack frame and defeats SROA (accumulators spill per MMA) -- see
        KB kernels/apple-m5-int8-matmul.

        `bounded=True`: ragged M/N tiles (+ K-tail) via the bounded MMA path.
        `bounded=False`: M/N-aligned -- full strips take width-16 when `k16` else
        masked width-4, K-tail always masked. `k16` is branched outside the loop.
        """
        comptime if bounded:
            for ks in range(n_full_strips):
                var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                mma_op.mma[bounded=True](
                    accum,
                    a_strip,
                    b_strip,
                    a_valid_rows=valid_rows,
                    b_valid_cols=valid_cols,
                    k_valid=Self.BK,
                )
            if has_k_tail:
                var ks = n_full_strips
                var k_rem = k - ks * Self.BK
                var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                mma_op.mma[bounded=True](
                    accum,
                    a_strip,
                    b_strip,
                    a_valid_rows=valid_rows,
                    b_valid_cols=valid_cols,
                    k_valid=k_rem,
                )
        else:
            if k16:
                comptime if Self.TTI32:
                    # int32-indexed A/B views; enqueue only dispatches here for
                    # shapes that fit i32, so no per-tile guard is needed.
                    var a32 = TileTensor[
                        .int8,
                        type_of(a).LayoutType,
                        type_of(a).origin,
                        Engine=type_of(a).Engine,
                        address_space=type_of(a).address_space,
                        linear_idx_type=.int32,
                    ](a._storage, a.layout)
                    var b32 = TileTensor[
                        .int8,
                        type_of(b).LayoutType,
                        type_of(b).origin,
                        Engine=type_of(b).Engine,
                        address_space=type_of(b).address_space,
                        linear_idx_type=.int32,
                    ](b._storage, b.layout)
                    for ks in range(n_full_strips):
                        Self._mma_width16_abs(
                            accum,
                            a32,
                            b32,
                            sg_row,
                            sg_col,
                            ks,
                            Int(mma_op.rb),
                            Int(mma_op.cb),
                        )
                else:
                    for ks in range(n_full_strips):
                        var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                        var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                        Self._mma_width16(
                            accum,
                            a_strip,
                            b_strip,
                            Int(mma_op.rb),
                            Int(mma_op.cb),
                        )
            else:
                for ks in range(n_full_strips):
                    var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                    var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                    Self._mma_masked(
                        accum,
                        a_strip,
                        b_strip,
                        Int(mma_op.rb),
                        Int(mma_op.cb),
                        valid_rows,
                        valid_cols,
                        Int(Self.BK),
                    )
            # K-tail: masked width-4, decoupled so the interior stays fast.
            if has_k_tail:
                var ks = n_full_strips
                var k_rem = k - ks * Self.BK
                var a_strip = a.tile[Self.SG_M, Self.BK](sg_row, ks)
                var b_strip = b.tile[Self.SG_N, Self.BK](sg_col, ks)
                Self._mma_masked(
                    accum,
                    a_strip,
                    b_strip,
                    Int(mma_op.rb),
                    Int(mma_op.cb),
                    valid_rows,
                    valid_cols,
                    k_rem,
                )

    @__name(t"apple_int8_matmul_run_{Self.c_type}_{Self.has_bias}")
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        as_layout: TensorLayout,
        bs_layout: TensorLayout,
        bias_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
        as_engine: TensorEngine,
        bs_engine: TensorEngine,
        bias_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[.int8, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[.int8, b_layout, ImmutAnyOrigin, Engine=b_engine],
        a_scale: TileTensor[
            .float32, as_layout, ImmutAnyOrigin, Engine=as_engine
        ],
        b_scale: TileTensor[
            .float32, bs_layout, ImmutAnyOrigin, Engine=bs_engine
        ],
        bias: TileTensor[
            Self.c_type, bias_layout, ImmutAnyOrigin, Engine=bias_engine
        ],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """W8A8 kernel entry. C `(M, N)`, A `(M, K)` int8, B `(N, K)` int8
        (`transpose_b`), `a_scale` `(M,)`, `b_scale` `(N,)`, `bias` `(N,)` (used
        iff `has_bias`). Grid is `(1<<log2_grid_m) * (1<<log2_grid_n)`
        threadgroups; OOB threadgroups early-return after Morton decode.

        Parameters:
            c_layout: `TensorLayout` of the C output `TileTensor`.
            a_layout: `TensorLayout` of the int8 A `TileTensor`.
            b_layout: `TensorLayout` of the int8 B `TileTensor`.
            as_layout: `TensorLayout` of the per-row activation scale
                `TileTensor`.
            bs_layout: `TensorLayout` of the per-column weight scale
                `TileTensor`.
            bias_layout: `TensorLayout` of the per-column bias `TileTensor`.
            c_engine: `TensorEngine` of `c`.
            a_engine: `TensorEngine` of `a`.
            b_engine: `TensorEngine` of `b`.
            as_engine: `TensorEngine` of `a_scale`.
            bs_engine: `TensorEngine` of `b_scale`.
            bias_engine: `TensorEngine` of `bias`.

        Args:
            c: Output `TileTensor` of shape `(M, N)` in `c_type`.
            a: Int8 activation `TileTensor` of shape `(M, K)`.
            b: Int8 weight `TileTensor` of shape `(N, K)` (used transposed).
            a_scale: Per-row fp32 activation scale, shape `(M,)`.
            b_scale: Per-column fp32 weight scale, shape `(N,)`.
            bias: Per-column bias in `c_type`, shape `(N,)` (used iff
                `has_bias`).
            log2_grid_m: Base-2 log of the power-of-two-padded M-side grid
                extent.
            log2_grid_n: Base-2 log of the power-of-two-padded N-side grid
                extent.
        """
        var m = Int32(c.dim[0]())
        var n = Int32(c.dim[1]())
        var k = Int(a.dim[1]())

        comptime BM_i = Int32(Self.BM)
        comptime BN_i = Int32(Self.BN)
        comptime SG_M_i = Int32(Self.SG_M)
        comptime SG_N_i = Int32(Self.SG_N)

        var grid_m = ceildiv(m, BM_i)
        var grid_n = ceildiv(n, BN_i)

        var tile_mn = Self.morton_decode_2d_rect(
            UInt32(block_idx.x), log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var sg_id = Int32(thread_idx.x) // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // Int32(Self.NUM_SG_N)
        var sg_n_idx = sg_id % Int32(Self.NUM_SG_N)

        var row_base = tile_m * BM_i + sg_m_idx * SG_M_i
        var col_base = tile_n * BN_i + sg_n_idx * SG_N_i
        if row_base >= m or col_base >= n:
            return

        # Absolute simdgroup-subtile indices (in SG_M / SG_N units).
        var sg_row = Int(row_base // SG_M_i)
        var sg_col = Int(col_base // SG_N_i)

        var mma_op = Self.Mma()
        var accum = Self.Mma.AccumType(fill=SIMD[.int32, 8](0))

        var is_edge = (row_base + SG_M_i > m) or (col_base + SG_N_i > n)
        var n_full_strips = k // Self.BK
        var has_k_tail = (k % Self.BK) != 0
        var k16 = (k % 16) == 0
        var valid_rows = Int(min(SG_M_i, m - row_base))
        var valid_cols = Int(min(SG_N_i, n - col_base))

        # K-loop: M/N-aligned tiles take the fast path (see `_accumulate`);
        # genuinely ragged M/N tiles take the bounded path.
        if is_edge:
            Self._accumulate[True](
                accum,
                a,
                b,
                mma_op,
                sg_row,
                sg_col,
                valid_rows,
                valid_cols,
                n_full_strips,
                has_k_tail,
                k,
                k16,
            )
        else:
            Self._accumulate[False](
                accum,
                a,
                b,
                mma_op,
                sg_row,
                sg_col,
                valid_rows,
                valid_cols,
                n_full_strips,
                has_k_tail,
                k,
                k16,
            )

        # === Dequant epilogue ============================================== #
        # The C-output write (int32 accum -> dequant -> `c_type` store) is owned
        # by `Int8DequantWriter`; `run` just hands it each 16x16 fragment + its
        # tile origin. Dequant elem (r, c) = int32 * a_scale[r] * b_scale[c]
        # (+ bias[c]) -- see the writer for the fragment/lane layout.
        var rb = Int(mma_op.rb)
        var cb = Int(mma_op.cb)
        var writer = Int8DequantWriter[
            Self.c_type,
            c_layout,
            as_layout,
            bs_layout,
            bias_layout,
            c_engine,
            as_engine,
            bs_engine,
            bias_engine,
            has_bias=Self.has_bias,
        ](c, a_scale, b_scale, bias, Int(m), Int(n))

        # Hoist the bounds branch once: interior tiles take the unguarded
        # `write_frag_full`, edge tiles the guarded `write_frag`.
        if is_edge:
            comptime for mi in range(Self.NUM_MMA_M):
                comptime for ni in range(Self.NUM_MMA_N):
                    writer.write_frag(
                        accum[mi * Self.NUM_MMA_N + ni],
                        Int(row_base) + mi * Self.MMA_M,
                        Int(col_base) + ni * Self.MMA_N,
                        rb,
                        cb,
                    )
        else:
            comptime for mi in range(Self.NUM_MMA_M):
                comptime for ni in range(Self.NUM_MMA_N):
                    writer.write_frag_full(
                        accum[mi * Self.NUM_MMA_N + ni],
                        Int(row_base) + mi * Self.MMA_M,
                        Int(col_base) + ni * Self.MMA_N,
                        rb,
                        cb,
                    )


@always_inline
def enqueue_apple_int8_matmul[
    c_type: DType = .bfloat16,
    *,
    has_bias: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.int8, ...],
    b: TileTensor[.int8, ...],
    a_scale: TileTensor[.float32, ...],
    b_scale: TileTensor[.float32, ...],
    bias: TileTensor[c_type, ...],
    ctx: DeviceContext,
    _use_i32_override: Optional[Bool] = None,
) raises:
    """Enqueue the W8A8 matmul: `out = dequant(a @ b^T)`.

    `a` is int8 `(M, K)`, `b` int8 `(N, K)` (`transpose_b`), `a_scale` `(M,)`
    per-row, `b_scale` `(N,)` per-column, `bias` `(N,)` in `c_type` (used iff
    `has_bias`; pass a length-1 dummy otherwise). C is `(M, N)`.

    Parameters:
        c_type: Output element type (bf16 / fp16 / fp32).
        has_bias: If True, add `bias[col]` after dequant.

    Args:
        c: Output `TileTensor` of shape `(M, N)` in `c_type`, written
            mutable.
        a: Int8 activation `TileTensor` of shape `(M, K)`.
        b: Int8 weight `TileTensor` of shape `(N, K)` (used transposed).
        a_scale: Per-row fp32 activation scale, shape `(M,)`.
        b_scale: Per-column fp32 weight scale, shape `(N,)`.
        bias: Per-column bias in `c_type`, shape `(N,)` (used iff
            `has_bias`; pass a length-1 dummy otherwise).
        ctx: Device context used to enqueue the kernel.
        _use_i32_override: Optional internal override forcing (`True`) or
            disabling (`False`) the int32 accumulation path; auto-selected
            when `None`.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime MM = AppleM5Int8MatMul[c_type, has_bias=has_bias]

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(b.dim[0]()) == n, "B must be (N, K)")
    debug_assert(Int(b.dim[1]()) == k, "B K must match A K")
    # MmaOpApple narrows row strides to UInt16.
    debug_assert(
        k <= 65535 and n <= 65535,
        "Apple int8 matmul: K and N must fit in UInt16",
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

    comptime kernel_i64 = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scale).LayoutType,
        type_of(b_scale).LayoutType,
        type_of(bias).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scale).Engine,
        type_of(b_scale).Engine,
        type_of(bias).Engine,
    ]
    comptime kernel_i32 = AppleM5Int8MatMul[
        c_type, has_bias=has_bias, TTI32=True
    ].run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scale).LayoutType,
        type_of(b_scale).LayoutType,
        type_of(bias).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scale).Engine,
        type_of(b_scale).Engine,
        type_of(bias).Engine,
    ]

    # Gate on the int32-indexed extents (m*k, n*k) -- not m*n, since only A/B are
    # int32-indexed (C stays i64). Non-qualifying shapes use the i64 `_mma_width16`.
    var use_i32 = _use_i32_override.value() if _use_i32_override else (
        (k % 16 == 0) and (max(m * k, n * k) < (Int(1) << 31))
    )
    if use_i32:
        ctx.enqueue_function[kernel_i32](
            c,
            a.as_immut(),
            b.as_immut(),
            a_scale.as_immut(),
            b_scale.as_immut(),
            bias.as_immut(),
            log2_m,
            log2_n,
            grid_dim=(side_m * side_n),
            block_dim=(MM.THREADS_PER_BLOCK),
        )
    else:
        ctx.enqueue_function[kernel_i64](
            c,
            a.as_immut(),
            b.as_immut(),
            a_scale.as_immut(),
            b_scale.as_immut(),
            bias.as_immut(),
            log2_m,
            log2_n,
            grid_dim=(side_m * side_n),
            block_dim=(MM.THREADS_PER_BLOCK),
        )


# ===----------------------------------------------------------------------=== #
# Online per-row activation quantization (bf16 -> int8 + per-row fp32 scale)
# ===----------------------------------------------------------------------=== #


struct AppleInt8ActQuant[in_type: DType = .bfloat16, *, THREADS: Int = 64]:
    """Per-row symmetric-absmax int8 quantization of the activation.

    One threadgroup per row: the `THREADS` threads cooperatively reduce
    `absmax` over the row's K columns (strided), then quantize
    `q = roundeven(x * 127 / absmax)` and write the row's fp32 `scale =
    absmax / 127`. Mirrors `_quantize_a_block`'s scheme (symmetric absmax/127,
    no zero-point) but as a GPU row kernel. No SMEM reduction primitive is
    assumed; the absmax reduction is a two-pass strided scan (the row fits in
    L1/L2 and is re-read cheaply -- K <= 12288 for FLUX).

    Parameters:
        in_type: Element type of the input activation tensor (defaults to
            `bfloat16`).
        THREADS: Number of threads per threadgroup cooperating on the per-row
            reduction (defaults to 64).
    """

    @__name("apple_int8_act_quant")
    @staticmethod
    def run[
        q_layout: TensorLayout,
        a_layout: TensorLayout,
        s_layout: TensorLayout,
        q_engine: TensorEngine,
        a_engine: TensorEngine,
        s_engine: TensorEngine,
    ](
        q: TileTensor[.int8, q_layout, MutAnyOrigin, Engine=q_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        a_scale: TileTensor[.float32, s_layout, MutAnyOrigin, Engine=s_engine],
        K_arg: Int32,
    ):
        var K = Int(K_arg)
        var row = Int(block_idx.x)
        var tid = Int(thread_idx.x)

        # Pass 1: per-thread partial absmax over strided columns of this row,
        # then a threadgroup all-reduce via a tiny shared-scratch max.
        var local_max = Float32(0)
        var j = tid
        while j < K:
            var v = abs(a.load[width=1](Coord(row, j)).cast[DType.float32]())
            if v > local_max:
                local_max = v
            j += Self.THREADS

        var row_max = _threadgroup_max[Self.THREADS](local_max)

        var scale = row_max / 127.0 if row_max != 0.0 else Float32(0)
        var mult = 127.0 / row_max if row_max != 0.0 else Float32(0)
        if tid == 0:
            a_scale.store[width=1](Coord(row), Float32(scale))

        # Pass 2: quantize.
        j = tid
        while j < K:
            var qi = Int(
                round(a.load[width=1](Coord(row, j)).cast[.float32]() * mult)
            )
            q.store[width=1](Coord(row, j), Int8(qi))
            j += Self.THREADS


@always_inline
def _threadgroup_max[nthreads: Int](val: Float32) -> Float32:
    """Threadgroup all-reduce max over `nthreads` (one warp-multiple block).

    Writes each thread's value to SMEM, barriers, then thread 0's linear scan
    broadcasts the max back through SMEM. `nthreads` is small (64) so the linear
    reduction is cheap and needs no tree.
    """
    var s = unsafe_stack_allocation[nthreads, Float32, address_space=.SHARED]()
    var tid = Int(thread_idx.x)
    s[tid] = val
    barrier()
    if tid == 0:
        var m = s[0]
        for i in range(1, nthreads):
            if s[i] > m:
                m = s[i]
        s[0] = m
    barrier()
    return s[0]


@always_inline
def enqueue_apple_int8_quantize_activation[
    in_type: DType = .bfloat16,
](
    q: TileTensor[mut=True, .int8, ...],
    a: TileTensor[in_type, ...],
    a_scale: TileTensor[mut=True, .float32, ...],
    ctx: DeviceContext,
) raises:
    """Quantize `a` `(M, K)` to int8 `q` + per-row fp32 `a_scale` `(M,)`.

    One threadgroup per row (M threadgroups, 64 threads each). Symmetric
    absmax/127, no zero-point. Raises if the GPU is not Apple M5.

    Parameters:
        in_type: Element type of the input activation tensor (defaults to
            `bfloat16`).

    Args:
        q: Output int8 `TileTensor` of shape `(M, K)`, written mutable.
        a: Input activation `TileTensor` of shape `(M, K)` in `in_type`.
        a_scale: Output per-row fp32 scale `TileTensor` of shape `(M,)`,
            written mutable.
        ctx: Device context used to enqueue the kernel.
    """
    _require_apple_m5(ctx)
    var m = Int(a.dim[0]())
    var k = Int(a.dim[1]())
    comptime QK = AppleInt8ActQuant[in_type]
    comptime kernel = QK.run[
        type_of(q).LayoutType,
        type_of(a).LayoutType,
        type_of(a_scale).LayoutType,
        type_of(q).Engine,
        type_of(a).Engine,
        type_of(a_scale).Engine,
    ]
    ctx.enqueue_function[kernel](
        q,
        a.as_immut(),
        a_scale,
        Int32(k),
        grid_dim=(m),
        block_dim=(QK.THREADS),
    )
