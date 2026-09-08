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
"""Apple Silicon MMA operation struct for TileTensor.

Simdgroup-level, register-owning MMA abstraction following the AMD MmaOp
pattern. Each simdgroup (32 threads) instantiates its own MmaOpApple.

Use mma() for interior tiles (caller guarantees in-bounds). Use
mma[bounded=True]() for edge tiles (zero-fills OOB elements). The
kernel should check once per simdgroup, not per load.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from max.gpu import lane_id
from max.gpu.compute.arch.mma_apple import _mma_apple_transposable
from max.gpu.memory import build_edge_mask, gmem_edge_masked_load
from std.math import divmod
from std.sys.info import align_of

from layout import TileTensor
from layout.coord import Coord


@fieldwise_init
struct ConvIm2colParams(
    Copyable, Deinitable, DevicePassable, ImplicitlyCopyable, Movable
):
    """Runtime conv geometry for the online im2col A-operand loader.

    SM100/Apple M5 (`compute_capability == 5`). The fused conv path
    (`AppleM5MatMul.run_conv`) never materialises the `[M, K]` im2col matrix;
    instead each A MMA-fragment is gathered directly from the NHWC input by the
    address math below. This struct carries the per-launch conv parameters the
    gather needs, and is `DevicePassable` (all-`Int32` POD, `device_type = Self`)
    so it can cross `enqueue_function` directly.

    The (M, K) -> NHWC map mirrors `nn/conv/gpu/im2col_matmul_2d.mojo` bit-for-bit
    so the fused result matches the materialised path:

      m -> (batch, h_out, w_out):  batch = m // HW_out;  spatial = m % HW_out;
                                    h_out = spatial // W_out; w_out = spatial % W_out
      k -> (r, s, c):              r = k // (S*C);  sc = k % (S*C);
                                    s = sc // C;  c = sc % C
      h_in = h_out*stride_h - pad_h + r;  w_in = w_out*stride_w - pad_w + s
      OOB (h_in/w_in outside [0,H)/[0,W)) -> 0
      in_idx = batch*(H*W*C) + h_in*(W*C) + w_in*C + c

    Dilation is assumed 1 (enforced at the conv dispatch gate); the map matches
    the materialiser, which also hardcodes dilation 1.
    """

    var H: Int32
    var W: Int32
    var C: Int32
    var R: Int32
    var S: Int32
    var H_out: Int32
    var W_out: Int32
    var pad_h: Int32
    var pad_w: Int32
    var stride_h: Int32
    var stride_w: Int32

    def __init__(out self):
        """Zero-init all fields, for the dense matmul path that ignores conv."""
        self.H = 0
        self.W = 0
        self.C = 0
        self.R = 0
        self.S = 0
        self.H_out = 0
        self.W_out = 0
        self.pad_h = 0
        self.pad_w = 0
        self.stride_h = 0
        self.stride_w = 0

    # `DevicePassable`: identity device form (all-Int32 POD, bit-copied).
    comptime device_type: AnyType = Self

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "ConvIm2colParams"


struct MmaOpApple[
    out_type: DType,
    in_type: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    *,
    b_type: DType = in_type,
    transpose_a: Bool = False,
    transpose_b: Bool = False,
]:
    """Apple Silicon simdgroup-level MMA abstraction for tile-based matrix multiply.

    Each simdgroup (32 threads) owns one instance and accumulates a
    `(num_m_mmas * 16) x (num_n_mmas * 16)` output tile using Apple simdgroup
    matrix multiply-accumulate instructions. Supports both dense and fused
    online-im2col A-operand paths.

    Parameters:
        out_type: Element type for the output accumulator (float32 or float16).
        in_type: Element type for the A operand.
        num_m_mmas: Number of 16x16 MMA tiles along the M dimension.
        num_n_mmas: Number of 16x16 MMA tiles along the N dimension.
        b_type: Element type for the B operand; defaults to in_type.
        transpose_a: When True, the A tile is stored in transposed (K, M) layout.
        transpose_b: When True, the B tile is stored in transposed (N, K) layout.
    """

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 16
    comptime FRAG_SIZE = 8
    comptime num_accum = Self.num_m_mmas * Self.num_n_mmas
    comptime AccumType = Array[
        SIMD[Self.out_type, Self.FRAG_SIZE], Self.num_accum
    ]

    var rb: Int
    var cb: Int

    @always_inline
    def __init__(out self):
        var lid = Int(lane_id())
        self.rb = ((lid & 7) >> 1) + ((lid & 16) >> 2)
        self.cb = ((lid & 1) << 2) + (lid & 8)

    @staticmethod
    @always_inline
    def zero_accum() -> Self.AccumType:
        return Self.AccumType(fill=SIMD[Self.out_type, Self.FRAG_SIZE](0))

    @staticmethod
    def _row_stride(tile: TileTensor) -> Int:
        """Return the non-unit stride (row stride for fragment layout).

        For row-major (stride[1]=1): returns stride[0].
        For col-major (stride[0]=1): returns stride[1].
        Asserts at comptime that exactly one stride is 1.
        """
        comptime if type_of(tile).static_stride[1] == 1:
            return Int(tile.layout.stride[0]().value())
        elif type_of(tile).static_stride[0] == 1:
            return Int(tile.layout.stride[1]().value())
        else:
            comptime assert (
                False
            ), "Tile must have a contiguous dimension (static_stride == 1)"

    @always_inline
    def _load_fragment[
        dtype: DType
    ](
        self,
        tile: TileTensor[dtype, ...],
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[
        dtype, 8
    ]:
        # Offsets are precomputed + hoisted once per slab in `mma()`.
        # Element alignment only: `rb * row_stride` is unaligned for odd K/N.
        # `width` drives vectorization, not the alignment hint.
        comptime alignment = align_of[Scalar[dtype]]()
        var lo = tile.raw_load[width=4, alignment=alignment](lo_off)
        var hi = tile.raw_load[width=4, alignment=alignment](hi_off)
        return lo.join(hi)

    @always_inline
    def _store_fragment[
        dtype: DType
    ](self, tile: TileTensor[mut=True, dtype, ...], frag: SIMD[dtype, 8],):
        var row_stride = Self._row_stride(tile)
        var off_lo = self.rb * row_stride + self.cb
        var off_hi = (self.rb + 8) * row_stride + self.cb
        tile.raw_store[width=4](off_lo, frag.slice[4, offset=0]())
        tile.raw_store[width=4](off_hi, frag.slice[4, offset=4]())

    @always_inline
    def load_fragment[
        dtype: DType, bounded: Bool = False
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int = 16,
    ) -> SIMD[
        dtype, 8
    ]:
        """Loads one 16x16 simdgroup fragment from a TileTensor sub-tile.

        The `_apple_frag_layout` bit-scatter is owned here at the `MmaOpApple`
        layer, not via a TileTensor `distribute` (KB
        `exceptions/apple-mma-fragment-is-not-distribute-expressible`). Used for
        the V operand of a P.V MMA, where A (P) is a register-resident score
        fragment and so cannot go through `mma()`.

        `bounded=True` zero-fills rows `>= valid_rows` instead of reading them
        -- needed for the V load on the last KV sub-tile when `num_keys % 16 !=
        0`, where reading an OOB row makes `0 * V_oob == NaN` and poisons the
        fp32 PV accumulator (KB `kernels/apple-m5-fa-prefill`). Depth columns are
        always in-bounds, so only rows are predicated.

        Parameters:
            dtype: The element dtype of the source tile.
            bounded: When True, zero-fill rows `>= valid_rows`.

        Args:
            tile: A 16x16 source view (row- or col-major).
            valid_rows: Valid rows from the sub-tile origin (only when bounded).

        Returns:
            This lane's 8-element fragment.
        """
        var row_stride = Self._row_stride(tile)
        var lo_off = self.rb * row_stride + self.cb
        var hi_off = (self.rb + 8) * row_stride + self.cb
        comptime if bounded:
            # All 16 cols valid (depth is contiguous within a token); only the
            # key (row) axis can run past the valid KV length.
            return self._bounded_load[dtype](
                tile, valid_rows, 16, lo_off, hi_off
            )
        else:
            return self._load_fragment[dtype](tile, lo_off, hi_off)

    @always_inline
    def _do_load[
        dtype: DType, bounded: Bool
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[dtype, 8]:
        """Dispatch fast or bounded load; swap bounds for col-major tiles."""
        comptime if bounded:
            comptime col_major = type_of(tile).static_stride[0] == 1
            comptime if col_major:
                return self._bounded_load[dtype](
                    tile, valid_cols, valid_rows, lo_off, hi_off
                )
            else:
                return self._bounded_load[dtype](
                    tile, valid_rows, valid_cols, lo_off, hi_off
                )
        else:
            return self._load_fragment[dtype](tile, lo_off, hi_off)

    @always_inline
    def _bounded_load[
        dtype: DType
    ](
        self,
        tile: TileTensor[dtype, ...],
        valid_rows: Int,
        valid_cols: Int,
        lo_off: Int,
        hi_off: Int,
    ) -> SIMD[dtype, 8]:
        """Bounded path: AGX3 edge-masked vector loads, zero-fill for OOB."""
        # Clamp to >= 0: a negative upper bound (e.g. valid_cols - ni*16 on
        # a small ragged tile) reads as unbounded in build_edge_mask.
        var col_mask = build_edge_mask(
            Int32(self.cb), Int32(0), Int32(max(0, valid_cols))
        )
        var lo_mask = col_mask if self.rb < valid_rows else Int16(0)
        var hi_mask = col_mask if self.rb + 8 < valid_rows else Int16(0)
        var lo = gmem_edge_masked_load[4](tile.ptr + lo_off, lo_mask)
        var hi = gmem_edge_masked_load[4](tile.ptr + hi_off, hi_mask)
        return lo.join(hi)

    @always_inline
    def _bounded_store[
        dtype: DType
    ](
        self,
        tile: TileTensor[mut=True, dtype, ...],
        frag: SIMD[dtype, 8],
        valid_rows: Int,
        valid_cols: Int,
    ):
        """Bounded path: predicated stores, skip OOB elements.

        Vectorized 4-wide when all 4 cols in bounds; scalar per-element
        when straddling the column boundary.
        """
        var row_stride = Self._row_stride(tile)
        var col = self.cb

        if self.rb < valid_rows:
            var off = self.rb * row_stride + col
            if col + 3 < valid_cols:
                tile.raw_store[width=4](off, frag.slice[4, offset=0]())
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        tile.raw_store[width=1](off + i, frag[i])

        if self.rb + 8 < valid_rows:
            var off = (self.rb + 8) * row_stride + col
            if col + 3 < valid_cols:
                tile.raw_store[width=4](off, frag.slice[4, offset=4]())
            else:
                for i in range(4):
                    if col + i < valid_cols:
                        tile.raw_store[width=1](off + i, frag[4 + i])

    @always_inline
    def mma[
        bounded: Bool = False
    ](
        self,
        mut accum: Self.AccumType,
        a_tile: TileTensor[Self.in_type, ...],
        b_tile: TileTensor[Self.b_type, ...],
        a_valid_rows: Int = Self.num_m_mmas * 16,
        b_valid_cols: Int = Self.num_n_mmas * 16,
        k_valid: Int = type_of(a_tile).static_shape[1],
    ):
        """Process K elements across all M/N tile positions.

        The K depth is inferred from the A tile's column dimension and
        must be a multiple of 16. For K=16 this is one MMA step; for
        K=32 this is two steps, etc. The struct iterates K internally.

        Tiles may be row-major or col-major. The stride layout is
        detected from static_stride and the hardware transpose flag is
        derived via XOR with the transpose parameter:
        hw_flag = is_col_major XOR transpose_param.

        Use mma() (bounded=False) for interior tiles where all memory
        is in-bounds. Use mma[bounded=True]() for edge tiles --
        zero-fills OOB elements. The kernel should check once per
        simdgroup, not per load.

        Parameters:
            bounded: When True, zero-fill out-of-bounds A/B elements
                instead of reading them (defaults to False).

        Args:
            accum: Caller-owned Array of SIMD[out_type, 8]
                accumulators, one per (num_m_mmas * num_n_mmas) tile.
            a_tile: A operand, shape (num_m_mmas * 16, K).
            b_tile: B operand, shape (K, num_n_mmas * 16) or
                (num_n_mmas * 16, K) if transpose_b.
            a_valid_rows: Valid rows from tile origin (bounded path only).
            b_valid_cols: Valid cols from tile origin (bounded path only).
            k_valid: Valid K elements across all steps (bounded path
                only). Defaults to the tile's full K dimension.
        """
        # Extract logical dimensions accounting for transpose.
        # A stored as (M, K) normally, (K, M) when transpose_a.
        # B stored as (K, N) normally, (N, K) when transpose_b.
        comptime a_k = (
            type_of(a_tile)
            .static_shape[0] if Self.transpose_a else type_of(a_tile)
            .static_shape[1]
        )
        comptime a_m = (
            type_of(a_tile)
            .static_shape[1] if Self.transpose_a else type_of(a_tile)
            .static_shape[0]
        )
        comptime b_k = (
            type_of(b_tile)
            .static_shape[1] if Self.transpose_b else type_of(b_tile)
            .static_shape[0]
        )
        comptime b_n = (
            type_of(b_tile)
            .static_shape[0] if Self.transpose_b else type_of(b_tile)
            .static_shape[1]
        )
        comptime assert a_k == b_k, "A and B K dimensions must match"
        comptime assert (
            a_k % Self.MMA_K == 0
        ), "K dimension must be a multiple of 16"
        comptime assert (
            a_m % Self.MMA_M == 0
        ), "A M dimension must be a multiple of 16"
        comptime assert (
            b_n % Self.MMA_N == 0
        ), "B N dimension must be a multiple of 16"

        # Hardware transpose = is_col_major XOR transpose_param.
        # _row_stride already asserts one stride is 1 at comptime.
        comptime a_col_major = type_of(a_tile).static_stride[0] == 1
        comptime b_col_major = type_of(b_tile).static_stride[0] == 1
        comptime hw_transpose_a = a_col_major != Self.transpose_a
        comptime hw_transpose_b = b_col_major != Self.transpose_b

        comptime num_k_steps = a_k // Self.MMA_K

        # Precompute the per-lane offsets once per slab.
        var a_row_stride = Self._row_stride(a_tile)
        var b_row_stride = Self._row_stride(b_tile)
        var rb_plus_8 = self.rb + 8
        var a_lo_off = self.rb * a_row_stride + self.cb
        var a_hi_off = rb_plus_8 * a_row_stride + self.cb
        var b_lo_off = self.rb * b_row_stride + self.cb
        var b_hi_off = rb_plus_8 * b_row_stride + self.cb

        comptime for ki in range(num_k_steps):
            # Pre-load B fragments for this K-step.
            var b_frags = Array[
                SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
            ](uninitialized=True)
            comptime for ni in range(Self.num_n_mmas):
                var b_sub = b_tile.tile[16, 16](
                    ni if Self.transpose_b else ki,
                    ki if Self.transpose_b else ni,
                )
                b_frags[ni] = self._do_load[Self.b_type, bounded](
                    b_sub,
                    (b_valid_cols - ni * 16) if Self.transpose_b else (
                        k_valid - ki * 16
                    ),
                    (k_valid - ki * 16) if Self.transpose_b else (
                        b_valid_cols - ni * 16
                    ),
                    b_lo_off,
                    b_hi_off,
                )

            comptime for mi in range(Self.num_m_mmas):
                var a_sub = a_tile.tile[16, 16](
                    ki if Self.transpose_a else mi,
                    mi if Self.transpose_a else ki,
                )
                var a_frag = self._do_load[Self.in_type, bounded](
                    a_sub,
                    (k_valid - ki * 16) if Self.transpose_a else (
                        a_valid_rows - mi * 16
                    ),
                    (a_valid_rows - mi * 16) if Self.transpose_a else (
                        k_valid - ki * 16
                    ),
                    a_lo_off,
                    a_hi_off,
                )

                comptime for ni in range(Self.num_n_mmas):
                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        a_frag,
                        b_frags[ni],
                        accum[mi * Self.num_n_mmas + ni],
                        hw_transpose_a,
                        hw_transpose_b,
                    )

    @always_inline
    def _load_frag_x2[
        dtype: DType, bounded: Bool
    ](
        self,
        tile: TileTensor[dtype, ...],
        lo_off: Int,
        hi_off: Int,
        two_cb: Int,
        valid_rows: Int,
        k_valid: Int,
    ) -> Tuple[SIMD[dtype, 8], SIMD[dtype, 8]]:
        """Load a BK=32 "double-strip" pair of fragments via FOUR width-4 loads.

        Four width-4 loads, not two width-8: a `load[width=8]` of `half`/`bf16`
        at element alignment SCALARIZES on the AGX backend (width-8 only
        vectorizes at `align 16`, which MISCOMPUTES on misaligned K). This is
        bit-identical to two separate BK=16 strips' width-4 loads, just grouped
        so the K-loop iterates half as often.
        """
        comptime align = align_of[Scalar[dtype]]()
        var lo_a: SIMD[dtype, 4]
        var lo_b: SIMD[dtype, 4]
        var hi_a: SIMD[dtype, 4]
        var hi_b: SIMD[dtype, 4]

        comptime if bounded:
            # Upper bound clamped >= 0, same reason as `_bounded_load`.
            var k_valid_c = Int32(max(0, k_valid))
            var mask_a = build_edge_mask(Int32(two_cb), Int32(0), k_valid_c)
            var mask_b = build_edge_mask(Int32(two_cb + 4), Int32(0), k_valid_c)
            var lo_mask_a = mask_a if self.rb < valid_rows else Int16(0)
            var lo_mask_b = mask_b if self.rb < valid_rows else Int16(0)
            var hi_mask_a = mask_a if self.rb + 8 < valid_rows else Int16(0)
            var hi_mask_b = mask_b if self.rb + 8 < valid_rows else Int16(0)
            lo_a = gmem_edge_masked_load[4](tile.ptr + lo_off, lo_mask_a)
            lo_b = gmem_edge_masked_load[4](tile.ptr + lo_off + 4, lo_mask_b)
            hi_a = gmem_edge_masked_load[4](tile.ptr + hi_off, hi_mask_a)
            hi_b = gmem_edge_masked_load[4](tile.ptr + hi_off + 4, hi_mask_b)
        else:
            lo_a = tile.raw_load[width=4, alignment=align](lo_off)
            lo_b = tile.raw_load[width=4, alignment=align](lo_off + 4)
            hi_a = tile.raw_load[width=4, alignment=align](hi_off)
            hi_b = tile.raw_load[width=4, alignment=align](hi_off + 4)

        return (lo_a.join(hi_a), lo_b.join(hi_b))

    @always_inline
    def mma_dense_x2[
        bounded: Bool = False
    ](
        self,
        mut accum: Self.AccumType,
        a_tile: TileTensor[Self.in_type, ...],
        b_tile: TileTensor[Self.b_type, ...],
        a_valid_rows: Int = Self.num_m_mmas * 16,
        b_valid_cols: Int = Self.num_n_mmas * 16,
        k_valid: Int = 32,
    ):
        """BK=32 "double-strip" dense NT MMA: process 32 K per call.

        A `(SG_M, 32)` row-major and B `(32, SG_N)` col-major are both K
        contiguous, so one call reads two BK=16 strips' worth of K -- halving
        the K-loop's iteration count -- via native `load[width=4]`s (see
        `_load_frag_x2`).

        Correctness: the `2*cb` split permutes K identically on both operands
        (`f0` = K-cols `{2cb..2cb+3}`, `f1` = `{2cb+4..2cb+7}`), and matmul
        contraction is K-permutation-invariant, so `f0`+`f1` sum to the full
        32-K contraction -- the same trick `mma_im2col`'s width-8 gather uses.

        Args:
            accum: Caller-owned accumulators (one per num_m_mmas * num_n_mmas).
            a_tile: A operand, `(num_m_mmas*16, 32)` row-major.
            b_tile: B operand, `(32, num_n_mmas*16)` col-major (NT).
            a_valid_rows: Valid M rows from the tile origin (bounded path only).
            b_valid_cols: Valid N cols from the tile origin (bounded path only).
            k_valid: Valid K elements in this 32-strip (partial-K tail only).
        """
        comptime a_k = type_of(a_tile).static_shape[1]
        comptime a_m = type_of(a_tile).static_shape[0]
        comptime b_k = type_of(b_tile).static_shape[0]
        comptime b_n = type_of(b_tile).static_shape[1]
        comptime assert a_k == 32, "mma_dense_x2 requires A K == 32"
        comptime assert b_k == 32, "mma_dense_x2 requires B K == 32"
        comptime assert a_m % 16 == 0, "A M dimension must be a multiple of 16"
        comptime assert b_n % 16 == 0, "B N dimension must be a multiple of 16"
        comptime assert (
            type_of(a_tile).static_stride[1] == 1
        ), "mma_dense_x2: A must be row-major (K contiguous)"
        comptime assert (
            type_of(b_tile).static_stride[0] == 1
        ), "mma_dense_x2: B must be col-major NT (K contiguous)"

        comptime hw_transpose_a = Self.transpose_a  # a_col_major == False
        comptime hw_transpose_b = not Self.transpose_b  # b_col_major == True

        var a_row_stride = Self._row_stride(a_tile)
        var b_row_stride = Self._row_stride(b_tile)
        var two_cb = 2 * self.cb

        # Preload both strips' B fragments per N sub-tile (as in `mma`).
        var b_frags0 = Array[
            SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
        ](uninitialized=True)
        var b_frags1 = Array[
            SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
        ](uninitialized=True)
        comptime for ni in range(Self.num_n_mmas):
            var r0 = ni * 16 + self.rb
            var lo_off = r0 * b_row_stride + two_cb
            var hi_off = (r0 + 8) * b_row_stride + two_cb
            var bf = self._load_frag_x2[Self.b_type, bounded](
                b_tile, lo_off, hi_off, two_cb, b_valid_cols - ni * 16, k_valid
            )
            b_frags0[ni] = bf[0]
            b_frags1[ni] = bf[1]

        comptime for mi in range(Self.num_m_mmas):
            var r0 = mi * 16 + self.rb
            var lo_off = r0 * a_row_stride + two_cb
            var hi_off = (r0 + 8) * a_row_stride + two_cb
            var af = self._load_frag_x2[Self.in_type, bounded](
                a_tile, lo_off, hi_off, two_cb, a_valid_rows - mi * 16, k_valid
            )
            comptime for ni in range(Self.num_n_mmas):
                _mma_apple_transposable(
                    accum[mi * Self.num_n_mmas + ni],
                    af[0],
                    b_frags0[ni],
                    accum[mi * Self.num_n_mmas + ni],
                    hw_transpose_a,
                    hw_transpose_b,
                )
                _mma_apple_transposable(
                    accum[mi * Self.num_n_mmas + ni],
                    af[1],
                    b_frags1[ni],
                    accum[mi * Self.num_n_mmas + ni],
                    hw_transpose_a,
                    hw_transpose_b,
                )

    @always_inline
    def _load_a_im2col_fragment_x2[
        input_origin: ImmOrigin, bounded: Bool, c_aligned: Bool, mi: Int
    ](
        self,
        input_ptr: UnsafePointer[Scalar[Self.in_type], input_origin],
        conv: ConvIm2colParams,
        h_base: Array[Int32, Self.num_m_mmas * 2],
        w_base: Array[Int32, Self.num_m_mmas * 2],
        batch_base: Array[Int32, Self.num_m_mmas * 2],
        c0: Int32,
        r: Int32,
        s: Int32,
        k_base: Int32,
        m_valid: Int32,
        k_total: Int32,
    ) -> Tuple[
        SIMD[Self.in_type, Self.FRAG_SIZE],
        SIMD[Self.in_type, Self.FRAG_SIZE],
    ]:
        """Width-8 im2col gather for a BK=32 strip: one width-8 load per
        row-half, split low-4 -> MMA0 / high-4 -> MMA1 (paired with
        `_load_b_fragment_x2`'s split). Anchors `(h_base, w_base, batch_base)`
        are prebaked and the K-decomposition `(c0, r, s)` carried by the loader,
        so the hot path has no `//`/`%`. Design + the A/B K-partition proof: KB
        `kernels/apple-conv2d-im2col`.
        """
        var SC = conv.S * conv.C
        var wc = conv.W * conv.C
        comptime in_align = align_of[Scalar[Self.in_type]]()

        var frag0 = SIMD[Self.in_type, Self.FRAG_SIZE](0)
        var frag1 = SIMD[Self.in_type, Self.FRAG_SIZE](0)

        var rb_i32 = Int32(self.rb)
        var k0base = k_base + Int32(2 * self.cb)

        # Row-half-invariant, so decide once and unswitch the loop. The
        # `comptime if not c_aligned` guard is why this is a comptime
        # specialization and not a plain runtime branch: under `c_aligned`
        # (C % 8 == 0) the channel term is never emitted, so `use_fast` folds to
        # constant `True` on the interior strips and the slow `else` path
        # dead-code-eliminates. The halo check below stays runtime regardless.
        var use_fast = True
        comptime if not c_aligned:
            use_fast = c0 + 7 < conv.C
        comptime if bounded:
            use_fast = use_fast and (k0base + 7 < k_total)

        if use_fast:
            # One width-8 load per row-half: the 8-channel run is contiguous.
            comptime for half in range(2):
                comptime out_off = half * 4
                comptime ri = mi * 2 + half
                comptime if bounded:
                    if rb_i32 + Int32(half * 8) >= m_valid:
                        continue
                var h_in = h_base[ri] + r
                var w_in = w_base[ri] + s
                if h_in >= 0 and h_in < conv.H and w_in >= 0 and w_in < conv.W:
                    var in_idx = batch_base[ri] + h_in * wc + w_in * conv.C + c0
                    var v = (input_ptr + Int(in_idx)).load[
                        width=8, alignment=in_align
                    ]()
                    comptime for i in range(4):
                        frag0[out_off + i] = v[i]
                        frag1[out_off + i] = v[4 + i]
        else:
            # Per-element gather: the run straddles a channel block (or the
            # partial-K tail), so each column maps to its own (r,s,c)/validity.
            comptime for half in range(2):
                comptime out_off = half * 4
                comptime ri = mi * 2 + half
                comptime if bounded:
                    if rb_i32 + Int32(half * 8) >= m_valid:
                        continue
                var h_anchor = h_base[ri]
                var w_anchor = w_base[ri]
                var batch_anchor = batch_base[ri]
                comptime for i in range(8):
                    var k = k0base + Int32(i)
                    comptime if bounded:
                        if k >= k_total:
                            continue
                    var r_i, sc_i = divmod(k, SC)
                    var s_i, c_i = divmod(sc_i, conv.C)
                    var h_in = h_anchor + r_i
                    var w_in = w_anchor + s_i
                    if (
                        h_in >= 0
                        and h_in < conv.H
                        and w_in >= 0
                        and w_in < conv.W
                    ):
                        var in_idx = (
                            batch_anchor + h_in * wc + w_in * conv.C + c_i
                        )
                        var val = input_ptr[Int(in_idx)]
                        comptime if i < 4:
                            frag0[out_off + i] = val
                        else:
                            frag1[out_off + (i - 4)] = val
        return (frag0, frag1)

    @always_inline
    def _load_b_fragment_x2[
        dtype: DType, bounded: Bool
    ](
        self,
        b_tile: TileTensor[dtype, ...],
        ni: Int,
        b_valid: Int,
        k_valid: Int,
    ) -> Tuple[SIMD[dtype, Self.FRAG_SIZE], SIMD[dtype, Self.FRAG_SIZE]]:
        """Width-8 B load for a BK=32 strip: read 8 contiguous K once, split
        low-4 -> MMA0 / high-4 -> MMA1 with the same K-partition as the A gather.

        Design (why width-8 + the A/B partition): KB `kernels/apple-conv2d-im2col`.
        `b_valid`/`k_valid` are the valid N / K from the strip origin (edge and
        partial-K zero-fill).
        """
        comptime assert (
            not Self.transpose_b
        ), "width-8 B load expects the conv NK operand (MMA transpose_b=False)"
        # (K-full, N-block) sub-tile; K is the contiguous (stride-1) axis.
        var b_msub = b_tile.tile[32, 16](0, ni)
        var row_stride = Self._row_stride(b_msub)
        var two_cb = 2 * self.cb
        var lo_off = self.rb * row_stride + two_cb
        var hi_off = (self.rb + 8) * row_stride + two_cb
        comptime align = align_of[Scalar[dtype]]()

        var lo8 = SIMD[dtype, 8](0)
        var hi8 = SIMD[dtype, 8](0)
        comptime if bounded:
            if self.rb < b_valid:
                if two_cb + 7 < k_valid:
                    lo8 = b_msub.raw_load[width=8, alignment=align](lo_off)
                else:
                    for i in range(8):
                        if two_cb + i < k_valid:
                            lo8[i] = b_msub.raw_load[width=1](lo_off + i)
            if self.rb + 8 < b_valid:
                if two_cb + 7 < k_valid:
                    hi8 = b_msub.raw_load[width=8, alignment=align](hi_off)
                else:
                    for i in range(8):
                        if two_cb + i < k_valid:
                            hi8[i] = b_msub.raw_load[width=1](hi_off + i)
        else:
            # Interior full strip: every N-col and all 32 K are in-bounds.
            lo8 = b_msub.raw_load[width=8, alignment=align](lo_off)
            hi8 = b_msub.raw_load[width=8, alignment=align](hi_off)

        var f0 = lo8.slice[4, offset=0]().join(hi8.slice[4, offset=0]())
        var f1 = lo8.slice[4, offset=4]().join(hi8.slice[4, offset=4]())
        return (f0, f1)

    @always_inline
    def mma_im2col[
        input_origin: ImmOrigin, bounded: Bool = True, c_aligned: Bool = False
    ](
        self,
        mut accum: Self.AccumType,
        input_ptr: UnsafePointer[Scalar[Self.in_type], input_origin],
        conv: ConvIm2colParams,
        b_tile: TileTensor[Self.b_type, ...],
        h_base: Array[Int32, Self.num_m_mmas * 2],
        w_base: Array[Int32, Self.num_m_mmas * 2],
        batch_base: Array[Int32, Self.num_m_mmas * 2],
        c0: Int32,
        r: Int32,
        s: Int32,
        k_base: Int32,
        m_valid: Int32,
        k_total: Int32,
        b_valid_cols: Int,
        k_valid: Int,
    ):
        """Fused-conv MMA step: A from online im2col gather, B as in `mma`.

        The A row anchors `(h_base, w_base, batch_base)` are prebaked by the
        loader (K-invariant `m -> pixel` decomposition, hoisted out of the
        K-loop); indexed per `(mi, half)` as `ri = mi*2 + half`. The strip's
        K-decomposition `(c0, r, s)` (= `k0base` -> tap/channel) is carried by
        the loader and passed in, so the gather does no `//`/`%` on the K axis.
        Apple M5 has no LDS stage, so the A operand is gathered per fragment
        rather than staged. Design: KB `kernels/apple-conv2d-im2col`.

        Parameters:
            input_origin: Memory origin of the NHWC input pointer (inferred).
            bounded: When True, zero-fill out-of-bounds A/B elements
                instead of reading them (defaults to True).
            c_aligned: When True, assume `conv.C` is a multiple of 8 so
                the channel run is contiguous and a single width-8 load
                suffices (defaults to False).

        Args:
            accum: Caller-owned accumulators (one per num_m_mmas * num_n_mmas).
            input_ptr: NHWC input base pointer.
            conv: Conv geometry for the im2col gather.
            b_tile: B operand, `(K, num_n_mmas*16)` or `(num_n_mmas*16, K)` if
                transpose_b -- same contract as `mma`.
            h_base: Prebaked per-row input-row anchor, indexed `ri = mi*2+half`.
            w_base: Prebaked per-row input-col anchor.
            batch_base: Prebaked per-row batch offset into the NHWC input.
            c0: Carried channel base of this strip's K origin.
            r: Carried filter-row tap of this strip's K origin.
            s: Carried filter-col tap of this strip's K origin.
            k_base: Absolute K (R*S*C) column where this BK strip starts.
            m_valid: Valid M rows from the simdgroup origin (ragged-M edge).
            k_total: Total K extent (R*S*C); A cols at or past it zero-fill.
            b_valid_cols: Valid N cols from the B tile origin (edge zero-fill).
            k_valid: Valid K elements in this strip (partial-K tail zero-fill).
        """
        comptime b_k = (
            type_of(b_tile)
            .static_shape[1] if Self.transpose_b else type_of(b_tile)
            .static_shape[0]
        )
        comptime b_n = (
            type_of(b_tile)
            .static_shape[0] if Self.transpose_b else type_of(b_tile)
            .static_shape[1]
        )
        comptime assert (
            b_k % Self.MMA_K == 0
        ), "K dimension must be a multiple of 16"
        comptime assert (
            b_n % Self.MMA_N == 0
        ), "B N dimension must be a multiple of 16"

        comptime b_col_major = type_of(b_tile).static_stride[0] == 1
        comptime hw_transpose_b = b_col_major != Self.transpose_b
        # A is im2col `(M, K)` row-major, never transposed.
        comptime hw_transpose_a = False

        comptime num_k_steps = b_k // Self.MMA_K

        comptime if num_k_steps == 2:
            var b_frags0 = Array[
                SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
            ](uninitialized=True)
            var b_frags1 = Array[
                SIMD[Self.b_type, Self.FRAG_SIZE], Self.num_n_mmas
            ](uninitialized=True)
            comptime for ni in range(Self.num_n_mmas):
                var bf = self._load_b_fragment_x2[bounded=bounded](
                    b_tile, ni, b_valid_cols - ni * 16, k_valid
                )
                b_frags0[ni] = bf[0]
                b_frags1[ni] = bf[1]

            comptime for mi in range(Self.num_m_mmas):
                # One width-8 gather for both K-MMAs of this M sub-tile, using
                # the lane's prebaked anchors for this fragment's two row-halves
                # (selected inside via `ri = mi*2 + half`).
                var af = self._load_a_im2col_fragment_x2[
                    bounded=bounded, c_aligned=c_aligned, mi=mi
                ](
                    input_ptr,
                    conv,
                    h_base,
                    w_base,
                    batch_base,
                    c0,
                    r,
                    s,
                    k_base,
                    m_valid - Int32(mi * 16),
                    k_total,
                )
                comptime for ni in range(Self.num_n_mmas):
                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        af[0],
                        b_frags0[ni],
                        accum[mi * Self.num_n_mmas + ni],
                        hw_transpose_a,
                        hw_transpose_b,
                    )
                    _mma_apple_transposable(
                        accum[mi * Self.num_n_mmas + ni],
                        af[1],
                        b_frags1[ni],
                        accum[mi * Self.num_n_mmas + ni],
                        hw_transpose_a,
                        hw_transpose_b,
                    )
        else:
            # NOTE: the conv path is fixed at BK=32 (num_k_steps == 2) by
            # `enqueue_apple_conv2d`, so only the width-8 path above is reachable.
            # A different BK would need a per-ki gather restored (see git
            # history); fail at compile time rather than silently mis-tile.
            comptime assert (
                False
            ), "Apple fused conv requires BK=32 (num_k_steps == 2)"

    @always_inline
    def store(
        self,
        accum: Self.AccumType,
        d_tile: TileTensor[mut=True, Self.out_type, ...],
    ):
        """Store all accumulators to output tile (unconditional).

        Caller guarantees all elements are in-bounds.

        Args:
            accum: Caller-owned accumulators to write, one per
                (num_m_mmas * num_n_mmas) tile.
            d_tile: Output tile of shape (num_m_mmas * 16,
                num_n_mmas * 16).
        """
        comptime for mi in range(Self.num_m_mmas):
            comptime for ni in range(Self.num_n_mmas):
                var sub = d_tile.tile[16, 16](mi, ni)
                self._store_fragment(sub, accum[mi * Self.num_n_mmas + ni])

    @always_inline
    def store_bounded(
        self,
        accum: Self.AccumType,
        d_tile: TileTensor[mut=True, Self.out_type, ...],
        valid_rows: Int,
        valid_cols: Int,
    ):
        """Stores accumulators where `(row < valid_rows) and (col < valid_cols)`.

        Assumes row-major `d_tile`; for col-major, mirror `_do_load`'s swap.

        Args:
            accum: Caller-owned accumulators to write, one per
                (num_m_mmas * num_n_mmas) tile.
            d_tile: Row-major output tile of shape (num_m_mmas * 16,
                num_n_mmas * 16).
            valid_rows: Valid rows from the tile origin; rows at or past
                this are skipped.
            valid_cols: Valid cols from the tile origin; cols at or past
                this are skipped.
        """
        comptime for mi in range(Self.num_m_mmas):
            comptime for ni in range(Self.num_n_mmas):
                var sub = d_tile.tile[16, 16](mi, ni)
                self._bounded_store(
                    sub,
                    accum[mi * Self.num_n_mmas + ni],
                    valid_rows=valid_rows - mi * 16,
                    valid_cols=valid_cols - ni * 16,
                )

    @always_inline
    def load_accum(
        self,
        d_tile: TileTensor[Self.out_type, ...],
    ) -> Self.AccumType:
        """Inverse of `store`: seed an accumulator from a previously-written
        output tile (chained-accumulate split-K, e.g. `clamp_v2`'s second
        pass, instead of a separate partials buffer + reduce kernel).

        Caller guarantees all elements are in-bounds.
        """

        def fragment_at[
            idx: Int
        ]() {imm} -> SIMD[Self.out_type, Self.FRAG_SIZE]:
            comptime mi, ni = divmod(idx, Self.num_n_mmas)
            return self.load_fragment[Self.out_type](
                d_tile.tile[16, 16](mi, ni)
            )

        return Self.AccumType(fill_with_unrolled=fragment_at)
