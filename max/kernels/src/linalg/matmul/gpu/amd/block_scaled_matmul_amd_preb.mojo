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
"""Block-scaled CDNA4 matmul with preshuffled B + scales + direct VGPR loads.

Variant of `BlockScaledMatmulAMD` that skips LDS staging for both B and the
A/B scales. B is preshuffled host-side via `Shuffler.preshuffle_b_5d`
so each lane's 16-byte fragment lives at a known DRAM offset and is
read with a single `buffer_load_dwordx4`. Scales are addressed by
`Shuffler.scale_4d_byte_off`: each lane reads one Int32 covering a
(mn_pack=2, k_pack=2) cell that feeds 4 sub-MMAs via the MFMA's
OPSEL byte selector.

Only suitable when `num_warps_m == 1` (BM == WM); otherwise B would be
read multiply across the warps in the M direction without LDS reuse.

Tile constraints:
  * `BM == 16 or BM % 32 == 0`. BM=16 uses one sub-MMA per CTA along M;
    the scale i32's mn_pack=1 byte is rotated into OPSEL byte 0/2 with
    `shrui` (see `BlockScaledMmaOp_PreB.mma`).
  * `WN == 16 or WN % 32 == 0`. Same logic per-warp along N.
  * `num_k_mmas` must be even (k_pack=2 cell halves).
  * `N` must be a multiple of 32 (= 16 * mn_pack) for B-scale cell alignment.
"""

from std.math import ceildiv
from std.math.uutils import udivmod
from std.memory.unsafe import bitcast
from std.sys import simd_width_of
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.host.info import MI355X
from max.gpu.memory import CacheOperation
from max.gpu.sync import s_waitcnt
from std.sys.intrinsics import llvm_intrinsic

from layout import Coord, TensorLayout, TensorEngine, TileTensor
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation
from layout.swizzle import Swizzle
from std.bit import log2_floor

from std.utils import StaticTuple, IndexList, StaticTuple
from linalg.arch.amd.block_scaled_mma import (
    CDNA4F8F6F4MatrixFormat,
    cdna4_block_scaled_mfma,
)
from structured_kernels.amd_tile_io import (
    RegTileLoader,
    RegTileWriter,
    TileLoaderLDS,
)

from .block_scaled_matmul_amd import MX_BLOCK_SIZE
from .block_scaled_preshuffle_layouts import Shuffler
from .block_scaled_preshuffle_loaders import (
    PreshuffledBLoader,
    PreshuffledScaleLoader,
)


@always_inline
def _lds_load_width[BK_BYTES: Int, BM: Int, max_width: Int]() -> Int:
    """Widest packed DMA width whose warp-tiling divides `BM`.

    `TileLoaderLDS` gives each warp `(WARP_SIZE * load_width) // BK_BYTES`
    rows, and that count must divide `BM` or the last warp-tile runs past the
    A tile. A power-of-two `BK_BYTES` tiles cleanly at the natural 16-byte
    width, so FP4 is unaffected; FP6's 192-byte tile yields 5 rows per warp at
    16 bytes -- 5 divides neither 64 nor 16 -- and falls back to 4, which
    yields 1 row per warp and always divides.

    Candidates deliberately EXCLUDE 12. The intrinsic accepts a 12-byte
    (dwordx3) size and it tiles 192 bytes neatly on paper, but the hardware
    writes those 12 bytes at a 16-byte lane pitch, leaving a 4-byte hole per
    lane -- so it is a padded-destination primitive, not a packed copy, and a
    caller assuming a 12-byte pitch interleaves every lane.
    Using it would require
    a padded LDS tile and a hole-skipping reader; dropping to 4 bytes keeps
    the tile packed at the cost of narrower DMA on the opt-in `dram_to_lds`
    path only.

    Parameters:
        BK_BYTES: Width of the A tile in bytes.
        BM: Height of the A tile in rows.
        max_width: Natural load width to start from, in bytes.

    Returns:
        The chosen load width in bytes.
    """
    comptime PACKED = StaticTuple[Int, 4](16, 4, 2, 1)
    comptime for i in range(len(PACKED)):
        comptime w = PACKED[i]
        comptime if w <= max_width:
            comptime rows_per_warp = (WARP_SIZE * w) // BK_BYTES
            comptime if (
                BK_BYTES % w == 0
                and rows_per_warp >= 1
                and BM % rows_per_warp == 0
            ):
                return w
    return 1


@always_inline
def _largest_divisor_at_most[n: Int](cap: Int) -> Int:
    """Returns the largest divisor of `n` that is at most `cap`.

    Parameters:
        n: The value to divide, positive.

    Args:
        cap: Upper bound on the returned divisor, positive.

    Returns:
        The largest `d` with `n % d == 0` and `d <= cap`.
    """
    var d = min(cap, n)
    while n % d != 0:
        d -= 1
    return d


# `swizzle_xor16`: XOR-16 LDS swizzle on a row-major [BM, BK_BYTES]
# uint8 A tile. The in-tile byte address is `row*BK_BYTES + (col ^ ((row & (BK_BYTES//16-1))*16))`
# — a 16B-granule XOR that removes A LDS bank conflicts. Applied identically at
# write (copy_a_tile_to_smem) and read (load_a_frag_from_smem); a mismatch is
# wrong logits. As a `Swizzle` on the flat in-tile byte offset: extract the
# `log2(BK_BYTES//16)` row bits sitting at flat-bit `log2(BK_BYTES)` and XOR
# them down into col's 16B-granule bits (base=4). yyy at base+shift => shift =
# log2(BK_BYTES)-4 = log2(BK_BYTES//16) = bits. BK_BYTES is pow-2 (64/128/256).
@always_inline
def a_lds_swizzle[BK_BYTES: Int]() -> Swizzle:
    """Builds the XOR-16 LDS swizzle for a row-major [BM, BK_BYTES] uint8 A tile.

    Returns a `Swizzle` on the flat in-tile byte offset that XORs the
    `log2(BK_BYTES//16)` row bits sitting at flat-bit `log2(BK_BYTES)` down
    into the column's 16B-granule bits (base=4), removing A LDS bank
    conflicts. Applied identically at write (`copy_a_tile_to_smem`) and read
    (`load_a_frag_from_smem`); a mismatch produces wrong logits. `BK_BYTES`
    must be a power of two (64/128/256).

    Parameters:
        BK_BYTES: Width of the A tile in bytes, a power of two.

    Returns:
        The XOR-16 `Swizzle` for the A LDS tile.
    """
    comptime granules = BK_BYTES // 16
    comptime if granules & (granules - 1) != 0:
        return Swizzle(0, 4, 0)
    comptime bits = log2_floor(granules)
    return Swizzle(bits, 4, bits)


# ===----------------------------------------------------------------------=== #
# BlockScaledMmaOp_PreB — preb-specific MFMA op with preshuffled-scale loads.
# ===----------------------------------------------------------------------=== #
#
# Sibling of `BlockScaledMmaOp` (the SMEM-scale variant in block_scaled_matmul_amd.mojo).
# Same A/B/C register storage and fragment loaders; differs only in:
#   * Scale storage shape: `[ceildiv(num_*_mmas, 2), num_k_mmas / 2]` — one
#     Int32 cell per (mn_pair, k_pair), covering up to 4 sub-MMAs at OPSEL
#     byte indices `(mma_k_idx % 2) * 2 + (m % 2)` (A) and same with `n` (B).
#     When num_*_mmas is odd (WM/WN=16), the last cell's mn_pack=1 byte is
#     unused; per-CTA `shrui` rotates the i32 in `mma()` so OPSEL byte 0/2
#     still picks the right scale.
#   * Scale loads come from `PreshuffledScaleLoader` (direct DRAM → VGPR),
#     not SMEM.
#
# Kept separate from `BlockScaledMmaOp` for readability while the
# preshuffled-scales path is being developed; can consolidate later.


struct BlockScaledMmaOp_PreB[
    mma_shape: IndexList[3],  # (16, 16, 128) for MXFP4 and MXFP8 alike
    warp_tile: IndexList[3],  # (WM, WN, BK_ELEMS) in MFMA-native element units
    num_b_slots: Int = 1,
    num_scale_slots: Int = 1,
    scale_group: Int = 1,
    b_addr_split: Bool = False,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
]:
    """Per-warp register state + MFMA dispatch for the preb (preshuffled-B,
    preshuffled-scale) kernel.

    `warp_tile` is the (M, N, K) region this warp computes per outer-K
    iteration, in the same element units as `mma_shape`. Per-warp MFMA
    counts are derived as `warp_tile[i] // mma_shape[i]`.

    Asserted in `__init__`: `warp_tile[i] % mma_shape[i] == 0` per axis,
    and `num_k_mmas % 2 == 0` (k_pack=2 cell halves). `num_m_mmas` /
    `num_n_mmas` may be odd; the constructor rotates the scale i32 per
    CTA so OPSEL keeps the same comptime formula. See module-level
    comment for the scale-cell byte ordering.

    Parameters:
        mma_shape: The MFMA instruction shape as `(M, N, K)` in
            MFMA-native element units, for example `(16, 16, 128)` for
            MXFP4.
        warp_tile: The `(M, N, K)` region this warp computes per
            outer-K iteration, in the same element units as
            `mma_shape`. Per-warp MFMA counts are derived as
            `warp_tile[i] // mma_shape[i]`.
        num_b_slots: Number of `_b_reg` slots for software pipelining
            (defaults to 1). Set to 2 to double-buffer B fragments
            across outer-K iterations.
        num_scale_slots: Number of A/B scale-register ring slots for
            software pipelining (defaults to 1). At the depth-2 default
            this is 1 (1-deep, unchanged behavior); the depth-3
            co-deepened prefetch sets it to 2 so the scale ring
            double-buffers alongside the B fragments.
        scale_group: Outer-K tiles whose scale atoms are fetched by one
            wide VMEM op (1 = off, per-tile `buffer_load_dword`).
        b_addr_split: Whether to split the non-FP6 B-fragment address into
            a loop-invariant per-lane part and a wave-uniform whole-tile
            part (defaults to False, one address per fragment).
        matrix_format: `f8f6f4` operand encoding for A and B. A lane covers
            32 K-elements in every format; the bytes that occupies -- 16
            (FP4), 24 (FP6), 32 (FP8) -- is derived from it.
    """

    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]

    comptime a_bits = Self.matrix_format.bits_per_element()
    comptime b_bits = Self.matrix_format.bits_per_element()
    comptime bits_per_element = Self.a_bits

    comptime num_m_mmas = Self.warp_tile[0] // Self.MMA_M
    comptime num_n_mmas = Self.warp_tile[1] // Self.MMA_N
    comptime num_k_mmas = Self.warp_tile[2] // Self.MMA_K

    comptime A_MMA_K_BYTES = (Self.MMA_K * Self.a_bits) // 8
    comptime B_MMA_K_BYTES = (Self.MMA_K * Self.b_bits) // 8
    comptime MMA_K_BYTES = Self.A_MMA_K_BYTES
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE  # 4
    comptime lanes_per_row = WARP_SIZE // Self.MMA_M
    comptime a_frag_width_bytes: Int = (
        Self.A_MMA_K_BYTES // Self.lanes_per_row
    )
    comptime b_frag_width_bytes: Int = (
        Self.B_MMA_K_BYTES // Self.lanes_per_row
    )
    comptime mma_frag_width_bytes: Int = Self.a_frag_width_bytes
    comptime a_reg_frag_bytes: Int = Self.matrix_format.simd_width()
    comptime b_reg_frag_bytes: Int = Self.matrix_format.simd_width()
    comptime reg_frag_bytes: Int = Self.a_reg_frag_bytes
    comptime lane_bytes: Int = Self.mma_frag_width_bytes
    # At MXFP8 a lane's two 16-byte halves are K_HALF_STRIDE apart in K, not
    # contiguous; treating them as contiguous mis-pairs the scale blocks.
    comptime FRAG_HALF_BYTES = 16
    comptime a_num_frag_halves = (
        Self.a_frag_width_bytes // Self.FRAG_HALF_BYTES
    )
    comptime b_num_frag_halves = (
        Self.b_frag_width_bytes // Self.FRAG_HALF_BYTES
    )
    comptime num_frag_halves = Self.a_num_frag_halves
    comptime A_K_HALF_STRIDE = Self.A_MMA_K_BYTES // Self.a_num_frag_halves
    comptime B_K_HALF_STRIDE = Self.B_MMA_K_BYTES // Self.b_num_frag_halves
    comptime K_HALF_STRIDE = Self.A_K_HALF_STRIDE
    # Per-slot A LDS tile is [BM, BK_BYTES].
    comptime BK_BYTES = (Self.warp_tile[2] * Self.a_bits) // 8

    comptime mx_format = Self.matrix_format

    comptime _a_reg_layout = row_major[
        Self.num_k_mmas,
        Self.num_m_mmas,
        Self.reg_frag_bytes,
    ]()
    # `_b_reg` holds B fragments for the current + (optionally) prefetched
    # outer-K iter. Stored 4D: [slot, mma_k_idx, n_mma, frag_bytes]. Slot
    # is outermost so the prefetch ring just toggles the leading index.
    comptime _b_reg_layout = row_major[
        Self.num_b_slots,
        Self.num_k_mmas,
        Self.num_n_mmas,
        Self.b_reg_frag_bytes,
    ]()
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
    ]()

    # 2x2 (mn_pack × k_pack) cell packing: one Int32 per (mi_pair, k_pair).
    # ceildiv so odd num_m_mmas/num_n_mmas (e.g. WM=16) still allocate a cell;
    # the unused mn_pack=1 byte is loaded but never OPSEL'd.
    # Slot-outermost so the S-deep scale ring toggles the leading index (1-deep default).
    comptime a_scale_packs = ceildiv(Self.num_m_mmas, 2)
    comptime b_scale_packs = ceildiv(Self.num_n_mmas, 2)
    comptime scale_packs = Self.a_scale_packs + Self.b_scale_packs
    comptime _a_scale_layout = row_major[
        Self.num_scale_slots, Self.a_scale_packs, Self.num_k_mmas // 2
    ]()
    comptime _b_scale_layout = row_major[
        Self.num_scale_slots, Self.b_scale_packs, Self.num_k_mmas // 2
    ]()

    # One 16 mn_lane x 4 k_lane atom of packed scale words; lane `l`'s
    # per-tile word sits at byte `packed_scale_bytes * l`. Derived from
    # `Shuffler`, not restated: the LDS strip geometry and `read_scale_group`'s
    # phase stride are only correct while this matches `scale_4d_byte_off`'s
    # own `bytes_per_atom`, and a disagreement would produce wrong scales
    # rather than a compile error.
    comptime SCALE_ATOM_BYTES = (
        Shuffler[1].MFMA_MN_LANES
        * Shuffler[1].MFMA_K_LANES
        * Shuffler[1].packed_scale_bytes
    )
    # Landing registers for the wide group fetch; one unused byte when off.
    comptime _scale_group_layout = row_major[
        Self.scale_packs if Self.scale_group > 1 else 1,
        Self.scale_group * 4 if Self.scale_group > 1 else 1,
    ]()

    var _a_reg: TileTensor[
        .uint8,
        type_of(Self._a_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_reg: TileTensor[
        .uint8,
        type_of(Self._b_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _c_reg: TileTensor[
        .float32,
        type_of(Self._c_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _a_scale_packed: TileTensor[
        .int32,
        type_of(Self._a_scale_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_scale_packed: TileTensor[
        .int32,
        type_of(Self._b_scale_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _scale_group_reg: TileTensor[
        .uint8,
        type_of(Self._scale_group_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    # Per-kernel runtime parity shifts for BM=16 / WN=16 cell-straddle.
    # 0 or 8 bits. Only consulted when warp_tile[0]==16 / warp_tile[1]==16;
    # otherwise the comptime-gated shift is eliminated.
    var _a_scale_shift: UInt32
    var _b_scale_shift: UInt32

    @always_inline
    def __init__(out self, warp_m_off: Int, warp_n_off: Int):
        comptime assert (
            Self.warp_tile[0] % Self.MMA_M == 0
        ), "warp_tile[0] (M) must be a multiple of mma_shape[0]"
        comptime assert (
            Self.warp_tile[1] % Self.MMA_N == 0
        ), "warp_tile[1] (N) must be a multiple of mma_shape[1]"
        comptime assert (
            Self.warp_tile[2] % Self.MMA_K == 0
        ), "warp_tile[2] (K) must be a multiple of mma_shape[2]"
        comptime assert (
            Self.num_k_mmas % 2 == 0
        ), "preb scale path requires num_k_mmas % 2 == 0 (k_pack=2)"

        self._a_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[DType.float32, address_space=.LOCAL](
            Self._c_reg_layout
        )
        _ = self._c_reg.fill(Float32(0))

        self._a_scale_packed = stack_allocation[
            DType.int32, address_space=.LOCAL
        ](Self._a_scale_layout)

        self._b_scale_packed = stack_allocation[
            DType.int32, address_space=.LOCAL
        ](Self._b_scale_layout)

        self._scale_group_reg = stack_allocation[
            DType.uint8, address_space=.LOCAL
        ](Self._scale_group_layout)

        # WM=16 / WN=16 cell-straddle: when warp_*_off // 16 is odd, the CTA's
        # m=0 / n=0 maps to the cell's mn_pack=1 byte. shrui by 8 in `mma()`
        # brings that byte to OPSEL position 0. For WM>=32 / WN>=32 parity is
        # always 0 and the shift is comptime-eliminated.
        self._a_scale_shift = UInt32(((warp_m_off >> 4) & 1) << 3)  # 0 or 8
        self._b_scale_shift = UInt32(((warp_n_off >> 4) & 1) << 3)

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_a_frag_from_smem[
        mma_k_idx: Int
    ](self, a_smem_warp: TileTensor[.uint8, _, _, address_space=.SHARED, ...],):
        """Loads the A fragment for MFMA-K position `mma_k_idx` from row-major SMEM.

        XOR-16 swizzled read (matches the write in `copy_a_tile_to_smem`): each
        lane reads the 16B vec at slot-tile (row, col_byte), then swizzles the
        flat in-tile byte offset before the `raw_load`. WM==BM so `a_smem_warp`
        IS the contiguous [BM, BK_BYTES] slot tile and `raw_load` indexes it
        directly.

        Parameters:
            mma_k_idx: Index of the MFMA step along K within the warp
                tile, in `[0, num_k_mmas)`.

        Args:
            a_smem_warp: The shared-memory A tile for this warp, a
                contiguous `[BM, BK_BYTES]` slot tile indexed directly
                by `raw_load`.
        """
        # col_major 16x4 lane layout: decode lane -> (m, k_vec).
        comptime lane_layout = col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M]()
        var crd = lane_layout.idx2crd(Int(lane_id()))
        var m = crd[0]
        var k_vec = crd[1]
        comptime swizzle = a_lds_swizzle[Self.BK_BYTES]()
        comptime tile_layout = row_major[Self.warp_tile[0], Self.BK_BYTES]()
        comptime if Self.bits_per_element == 6:
            comptime assert (
                Self.mma_frag_width_bytes % 8 == 0
            ), "FP6 fragment must be a whole number of 8-byte chunks"
            var a_reg_fp6 = self._a_reg.vectorize[1, 1, Self.reg_frag_bytes]()
            comptime for i in range(Self.num_m_mmas):
                var row = i * Self.MMA_M + Int(m)
                var col_byte = (
                    mma_k_idx * Self.MMA_K_BYTES
                    + Int(k_vec) * Self.mma_frag_width_bytes
                )
                var off = swizzle(Int(tile_layout(Coord(row, col_byte))))
                var frag = SIMD[.uint8, Self.reg_frag_bytes](0)
                comptime for chunk in range(Self.mma_frag_width_bytes // 8):
                    frag = frag.insert[offset=chunk * 8](
                        rebind[SIMD[.uint8, 8]](
                            a_smem_warp.raw_load[width=8](off + chunk * 8)
                        )
                    )
                a_reg_fp6[mma_k_idx, i, 0] = frag
        else:
            var a_reg_v = self._a_reg.vectorize[1, 1, Self.FRAG_HALF_BYTES]()
            comptime for i in range(Self.num_m_mmas):
                var row = i * Self.MMA_M + Int(m)
                comptime for h in range(Self.num_frag_halves):
                    var col_byte = (
                        mma_k_idx * Self.MMA_K_BYTES
                        + h * Self.K_HALF_STRIDE
                        + Int(k_vec) * Self.FRAG_HALF_BYTES
                    )
                    var off = swizzle(Int(tile_layout(Coord(row, col_byte))))
                    a_reg_v[mma_k_idx, i, h] = a_smem_warp.raw_load[
                        width=Self.FRAG_HALF_BYTES
                    ](off)

    @always_inline
    def load_b_frag_preshuffled[
        B_N: Int,
        B_K_BYTES: Int,
        b_cache: CacheOperation,
        b_lane_bytes: Int,
        //,
        mma_k_idx: Int,
        slot: Int = 0,
    ](
        self,
        b_loader: PreshuffledBLoader[B_N, B_K_BYTES, b_cache, b_lane_bytes],
        warp_n_off: Int,
        k_byte_base: Int,
    ):
        """Loads B fragments direct from preshuffled DRAM into b_reg slot `slot`.

        Parameters:
            mma_k_idx: Index of the MFMA step along K within the warp
                tile, in `[0, num_k_mmas)`.
            slot: The `_b_reg` slot to load into (defaults to 0).

        Args:
            b_loader: The `PreshuffledBLoader` for the preshuffled B
                tensor.
            warp_n_off: Global N offset of this warp's tile.
            k_byte_base: Base byte offset along K for the current
                outer-K tile.
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"

        var lane_klane, lane_nlane = udivmod(lane_id(), Self.MMA_N)

        var b_reg_v = self._b_reg.vectorize[1, 1, 1, Self.FRAG_HALF_BYTES]()
        comptime for i in range(Self.num_n_mmas):
            # the logical n row in the expert we will be loading from
            # the warp_n_offset is the tile base position for this warp block,
            # i * Self.MMA_N shifts down by 16 based on what MMA this warp is
            # processing in the warp tile, then we add the specific lane in n
            # this tile is responsible for.
            var n_log = warp_n_off + i * Self.MMA_N + lane_nlane

            # K_byte_base the starting byte offset based on the Kth block tile we are on.
            # mma_k_idx * Self.MMA_K_BYTES, shifts that based on the mma_k tile we are
            # processing in that block. Finally we add the lane's klane offset within that K tile,
            # This is usually a multiple of 16

            comptime if Self.b_bits == 6:
                var k_byte_log = (
                    k_byte_base
                    + mma_k_idx * Self.B_MMA_K_BYTES
                    + lane_klane * Self.b_frag_width_bytes
                )
                self._b_reg.vectorize[1, 1, 1, Self.b_reg_frag_bytes]()[
                    slot, mma_k_idx, i, 0
                ] = rebind[SIMD[.uint8, Self.b_reg_frag_bytes]](
                    b_loader.load_fragment(n_log, k_byte_log)
                )
            elif Self.b_addr_split:
                # Loop-invariant lane part into voffset, wave-uniform whole
                # tiles into soffset, hoisting the `v_add` chain out of the
                # mainloop. Safe under either buffer range-check convention
                # (soffset included or excluded): voffset alone carries the
                # whole `n0` stride, so an `n` past N already exceeds
                # num_records, and soffset only ever adds in-bounds K tiles
                # (`k0 < k0_count` for every iteration of the K loop).
                var lane_off = b_loader.lane_plane_off(
                    n_log, lane_klane * Self.FRAG_HALF_BYTES
                )
                comptime for h in range(Self.b_num_frag_halves):
                    var k_uniform = (
                        k_byte_base
                        + mma_k_idx * Self.B_MMA_K_BYTES
                        + h * Self.B_K_HALF_STRIDE
                    )
                    b_reg_v[slot, mma_k_idx, i, h] = rebind[
                        SIMD[.uint8, Self.FRAG_HALF_BYTES]
                    ](b_loader.load_at(lane_off, k_uniform))
            else:
                comptime for h in range(Self.b_num_frag_halves):
                    var k_byte_log = (
                        k_byte_base
                        + mma_k_idx * Self.B_MMA_K_BYTES
                        + h * Self.B_K_HALF_STRIDE
                        + lane_klane * Self.FRAG_HALF_BYTES
                    )
                    b_reg_v[slot, mma_k_idx, i, h] = rebind[
                        SIMD[.uint8, Self.FRAG_HALF_BYTES]
                    ](b_loader.load_fragment(n_log, k_byte_log))

    @always_inline
    def load_a_scales_preshuffled[
        k_pair: Int, slot: Int = 0
    ](
        mut self,
        a_scale_loader: PreshuffledScaleLoader[_, _],
        warp_m_off: Int,
        k_pair_idx: Int,
    ):
        """Issues per-lane i32 scale loads for A at one k_pair slot.

        Caller provides the absolute `k_pair_idx` (= `k_iter *
        (num_k_mmas / 2) + k_pair`); each step advances by 8 K-scales
        (= 2 MFMAs along K). One i32 per (mi_pair, k_pair) per lane.

        Parameters:
            k_pair: Index of the k_pair slot within the current outer-K
                tile, in `[0, num_k_mmas // 2)`.
            slot: Scale-ring slot (comptime; 0 at the 1-deep default).

        Args:
            a_scale_loader: The `PreshuffledScaleLoader` for the A
                scale tensor.
            warp_m_off: Global M offset of this warp's tile.
            k_pair_idx: Absolute k_pair index across all outer-K
                iterations, equal to `k_iter * (num_k_mmas / 2) +
                k_pair`; each step advances by 8 K-scales.
        """
        comptime assert k_pair < Self.num_k_mmas // 2, "k_pair out of range"
        comptime assert slot < Self.num_scale_slots, "scale slot out of range"

        var lane_klane, lane_mn = udivmod(lane_id(), Self.MMA_M)

        var k_scale_idx = k_pair_idx * 8 + lane_klane

        comptime for m_pack_idx in range(ceildiv(Self.num_m_mmas, 2)):
            var mn_log = warp_m_off + m_pack_idx * 32 + lane_mn
            self._a_scale_packed[
                slot, m_pack_idx, k_pair
            ] = a_scale_loader.load_packed(mn_log, k_scale_idx)

    @always_inline
    def load_b_scales_preshuffled[
        k_pair: Int, slot: Int = 0
    ](
        mut self,
        b_scale_loader: PreshuffledScaleLoader[_, _],
        warp_n_off: Int,
        k_pair_idx: Int,
    ):
        """Mirror of `load_a_scales_preshuffled` along N.

        Parameters:
            k_pair: Index of the k_pair slot within the current outer-K
                tile, in `[0, num_k_mmas // 2)`.
            slot: Scale-ring slot (comptime; 0 at the 1-deep default).

        Args:
            b_scale_loader: The `PreshuffledScaleLoader` for the B
                scale tensor.
            warp_n_off: Global N offset of this warp's tile.
            k_pair_idx: Absolute k_pair index across all outer-K
                iterations, equal to `k_iter * (num_k_mmas / 2) +
                k_pair`; each step advances by 8 K-scales.
        """
        comptime assert k_pair < Self.num_k_mmas // 2, "k_pair out of range"
        comptime assert slot < Self.num_scale_slots, "scale slot out of range"

        var lane_klane, lane_mn = udivmod(lane_id(), Self.MMA_N)

        var k_scale_idx = k_pair_idx * 8 + lane_klane

        comptime for n_pack_idx in range(ceildiv(Self.num_n_mmas, 2)):
            var mn_log = warp_n_off + n_pack_idx * 32 + lane_mn
            self._b_scale_packed[
                slot, n_pack_idx, k_pair
            ] = b_scale_loader.load_packed(mn_log, k_scale_idx)

    @always_inline
    def stage_scale_group(
        mut self,
        a_scale_loader: PreshuffledScaleLoader[_, _],
        b_scale_loader: PreshuffledScaleLoader[_, _],
        warp_m_off: Int,
        warp_n_off: Int,
        k_pair_base: Int,
    ):
        """Fetches `scale_group` outer-K tiles of A+B scales in one op per pack.

        A wave64 VMEM op costs the same whether it presents 4 or 16 bytes per
        lane, so this cuts scale VMEM instructions by `scale_group`.

        Issue it at the top of a tile and `publish_scale_group` at the bottom,
        so the A landing registers' existing `vmcnt` covers the fetch.

        Args:
            a_scale_loader: The `PreshuffledScaleLoader` for A scales.
            b_scale_loader: The `PreshuffledScaleLoader` for B scales.
            warp_m_off: Global M offset of this warp's tile.
            warp_n_off: Global N offset of this warp's tile.
            k_pair_base: First `k_pair` of the window.
        """
        comptime G = Self.scale_group
        comptime assert G > 1, "stage_scale_group needs scale_group > 1"
        var group_v = self._scale_group_reg.vectorize[1, G * 4]()

        comptime for p in range(Self.a_scale_packs):
            group_v[p, 0] = rebind[SIMD[.uint8, G * 4]](
                a_scale_loader.load_group[G](warp_m_off + p * 32, k_pair_base)
            )
        comptime for q in range(Self.b_scale_packs):
            group_v[Self.a_scale_packs + q, 0] = rebind[SIMD[.uint8, G * 4]](
                b_scale_loader.load_group[G](warp_n_off + q * 32, k_pair_base)
            )

    @always_inline
    def publish_scale_group(
        self,
        scale_smem: TileTensor[
            mut=True, .uint8, _, _, address_space=.SHARED, ...
        ],
        warp_id: Int,
    ):
        """Writes the staged window to this warp's LDS strip, one op per pack.

        `load_group` delivers the window lane-transposed; bouncing it through
        LDS undoes that far more cheaply than the VMEM ops it replaces.

        Args:
            scale_smem: The CTA's `[num_warps * scale_packs *
                scale_group, SCALE_ATOM_BYTES]` LDS staging strip.
            warp_id: This warp's index; selects its strip.
        """
        comptime G = Self.scale_group
        comptime assert G > 1, "publish_scale_group needs scale_group > 1"
        comptime PACK_STRIDE = G * Self.SCALE_ATOM_BYTES
        var group_v = self._scale_group_reg.vectorize[1, G * 4]()
        var base = warp_id * Self.scale_packs * PACK_STRIDE + Int(lane_id()) * (
            G * 4
        )

        comptime for p in range(Self.scale_packs):
            scale_smem.raw_store[width=G * 4](
                base + p * PACK_STRIDE,
                rebind[SIMD[.uint8, G * 4]](group_v[p, 0]),
            )

    @always_inline
    def read_scale_group[
        phase: Int, slot: Int = 0
    ](
        mut self,
        scale_smem: TileTensor[.uint8, _, _, address_space=.SHARED, ...],
        warp_id: Int,
    ):
        """Reads tile `phase` of the published window into the scale registers.

        Parameters:
            phase: Position of this outer-K tile inside the window, in
                `[0, scale_group)`.
            slot: Scale-ring slot to fill (0 at the 1-deep default).

        Args:
            scale_smem: The CTA's LDS staging strip, already published.
            warp_id: This warp's index; selects its strip.
        """
        comptime G = Self.scale_group
        comptime assert G > 1, "read_scale_group needs scale_group > 1"
        comptime assert phase < G, "phase out of range"
        comptime assert slot < Self.num_scale_slots, "scale slot out of range"
        comptime assert (
            Self.num_k_mmas // 2 == 1
        ), "grouped scale loads assume one k_pair per outer-K tile"
        comptime PACK_STRIDE = G * Self.SCALE_ATOM_BYTES
        var off = (
            warp_id * Self.scale_packs * PACK_STRIDE
            + phase * Self.SCALE_ATOM_BYTES
            + Int(lane_id()) * 4
        )

        comptime for p in range(Self.a_scale_packs):
            self._a_scale_packed[slot, p, 0] = bitcast[.int32, 1](
                scale_smem.raw_load[width=4](p * PACK_STRIDE + off)
            )[0]
        comptime for q in range(Self.b_scale_packs):
            self._b_scale_packed[slot, q, 0] = bitcast[.int32, 1](
                scale_smem.raw_load[width=4](
                    (Self.a_scale_packs + q) * PACK_STRIDE + off
                )
            )[0]

    @always_inline
    def mma[mma_k_idx: Int, slot: Int = 0, scale_slot: Int = 0](self):
        """Block-scaled MFMA at MFMA-K position `mma_k_idx` using B from `slot`.

        B-major / n-outer / m-inner: hoist the B fragment + b_byte + b_scale
        (VMEM-loaded `_b_reg`) once per n, then cycle A (m-inner, LDS-loaded
        `_a_reg`). Keeping B resident across the m-loop improves MFMA ILP.

        OPSEL byte selection from the 2x2 cell:
            a_byte = (mma_k_idx % 2) * 2 + (m % 2)
            b_byte = (mma_k_idx % 2) * 2 + (n % 2)
        Scale dword lives at `_*_scale_packed[mn // 2, mma_k_idx // 2]`. WM/WN=16
        CTAs see only m=0 / n=0, so the constructor `shrui`
        (`_a_scale_shift` / `_b_scale_shift`) rotates the i32 to the right OPSEL
        byte.

        Parameters:
            mma_k_idx: Index of the MFMA step along K within the warp
                tile, in `[0, num_k_mmas)`.
            slot: The `_b_reg` slot to read B fragments from (defaults
                to 0).
            scale_slot: The scale-ring slot to read A/B scales from
                (defaults to 0; non-zero only under the depth-3
                co-deepened prefetch).
        """
        comptime assert slot < Self.num_b_slots, "slot out of range"
        comptime assert (
            scale_slot < Self.num_scale_slots
        ), "scale slot out of range"
        var a_reg_v = self._a_reg.vectorize[1, 1, Self.a_reg_frag_bytes]()
        var b_reg_v = self._b_reg.vectorize[1, 1, 1, Self.b_reg_frag_bytes]()
        var c_reg_v = self._c_reg.vectorize[1, Self.c_frag_size]()

        # 2x2 (k_pack × mn_pack) cell -> OPSEL byte: (k%2, mn%2) row-major.
        comptime scale_cell = row_major[2, 2]()

        comptime for n in range(Self.num_n_mmas):
            # B-side state — invariant across the inner m loop.
            var b_frag = b_reg_v[slot, mma_k_idx, n, 0]

            comptime b_byte = (mma_k_idx % 2) * 2 + (n % 2)
            var b_scale = rebind[Int32](
                self._b_scale_packed[scale_slot, n // 2, mma_k_idx // 2]
            )
            comptime if Self.warp_tile[1] == 16:
                b_scale = Int32(UInt32(b_scale) >> self._b_scale_shift)

            comptime for m in range(Self.num_m_mmas):
                var a_frag = a_reg_v[mma_k_idx, m, 0]

                var c_frag = c_reg_v[m, n]

                comptime a_byte = (mma_k_idx % 2) * 2 + (m % 2)
                var a_scale = rebind[Int32](
                    self._a_scale_packed[scale_slot, m // 2, mma_k_idx // 2]
                )
                comptime if Self.warp_tile[0] == 16:
                    a_scale = Int32(UInt32(a_scale) >> self._a_scale_shift)

                cdna4_block_scaled_mfma[
                    Int32(b_byte),
                    Int32(a_byte),
                    Self.matrix_format,
                    Self.matrix_format,
                ](
                    c_frag,
                    b_frag,
                    a_frag,
                    b_scale,
                    a_scale,
                )

                c_reg_v[m, n] = c_frag


struct BlockScaledMatmulAMD_PreB[
    BM: Int = 64,
    BN: Int = 128,
    BK_ELEMS: Int = 512,
    WN: Int = 64,
    b_prefetch: Bool = False,
    b_cache_policy: CacheOperation = CacheOperation.ALWAYS,
    dram_to_lds: Bool = False,
    cluster_drain_sched: Bool = False,
    mfma_cluster: Int = 4,
    pipeline_depth: Int = 2,
    scale_group: Int = 1,
    b_addr_split: Bool = False,
    matrix_format: CDNA4F8F6F4MatrixFormat = CDNA4F8F6F4MatrixFormat.FLOAT4_E2M1,
]:
    """Preshuffled-B variant of `BlockScaledMatmulAMD`.

    The preb path requires `num_warps_m == 1` (no LDS staging for B = no
    cross-warp M-direction B reuse), so `WM` is structurally fixed to `BM`.

    When `b_prefetch=True`, runs a depth-2 outer-K software pipeline: while
    the current iter's MFMAs execute, the next iter's B fragments stream
    from DRAM into the alternate b_reg slot. Doubles `_b_reg` size (extra
    VGPRs) but hides DRAM B latency across the inner MFMA chain. Targets
    K-heavy shapes (e.g. gate/up, K=7168) where outer-iter serialization
    dominates.

    `cluster_drain_sched` (b_prefetch only) switches the 1-deep steady loop to
    an interleaved B-issue schedule: the next tile's B fragments are issued
    per-k *between* the current tile's MFMA phases (not front-loaded), each phase
    pinned by `sched_barrier(0)` + bracketed by `s_setprio`, and the
    end-of-tile sync is a bare `s_barrier` + `lgkmcnt`-only drain so in-flight
    B DMAs cross it. (The epilogue fully drains `vmcnt` -- it has no newer ops
    to stage a staircase against -- and keeps the per-cluster `s_setprio`
    bracketing.) Default off: callers bit-identical unless opted in.

    `b_addr_split` splits the non-FP6 B-fragment address into a loop-invariant
    per-lane part (voffset) and a wave-uniform whole-tile part (soffset),
    hoisting the per-tile `v_add` chain out of the unrolled K loop. Worth
    3-4% on the wide MXFP8 gate+up tiles (BM 32..128) and *costs* 3-4% on the
    BM=16 MXFP4 decode tiles, where there is almost no K loop to amortise the
    scalar setup against -- so it is opted into per band, not global.

    `pipeline_depth` sizes the B-fragment register ring (`num_b_slots`) and,
    when > 2, switches the b_prefetch steady loop's end-of-iter sync to the
    non-draining `s_waitcnt[lgkmcnt=0]` + bare `s_barrier` so in-flight B DMAs
    are NOT drained every iteration. At the default (2) the ring is the same
    2 slots and the draining `barrier()` is kept — bit-identical to before.
    The deeper prefetch *schedule* that consumes slots >= 2 is a follow-up; this
    param only plumbs the depth + the non-draining-barrier seam.

    MFMA consumption order is B-major (n-outer / m-inner): the B fragment is
    held resident across the m-loop for better MFMA ILP. See `mma`.

    Parameters:
        BM: CTA tile size along M in elements, either 16 or a multiple
            of 32. `WM` is locked to `BM` (single warp along M).
        BN: CTA tile size along N in elements, split across
            `num_warps_n = BN // WN` warps.
        BK_ELEMS: K tile size in MXFP4 elements per outer-K iteration;
            must be a multiple of 256 so `num_k_mmas` is even.
            `BK_BYTES = BK_ELEMS // 2`.
        WN: Per-warp tile size along N in elements, either 16 or a
            multiple of 32.
        b_prefetch: Enables a depth-2 outer-K software pipeline that
            double-buffers B fragments across iterations (defaults to
            `False`).
        b_cache_policy: `CacheOperation` hint applied to preshuffled B
            DRAM loads (defaults to `CacheOperation.ALWAYS`).
        dram_to_lds: Routes A loads through the shared swizzled
            `TileLoaderLDS` DRAM-to-LDS path instead of a register
            bounce (defaults to `False`).
        cluster_drain_sched: Switches the prefetch steady loop to an
            interleaved B-issue schedule with per-cluster `s_setprio`
            and a partial-`vmcnt` staircase (defaults to `False`).
        mfma_cluster: Number of MFMAs per cluster in the scheduled
            MFMA chain used by `cluster_drain_sched` (defaults to 4).
        pipeline_depth: Depth of the B-fragment register ring
            (`num_b_slots`) on the prefetch path (defaults to 2). When
            `> 2` it also switches the steady loop's end-of-iter sync to
            the non-draining `s_waitcnt[lgkmcnt=0]` + bare `s_barrier`
            and co-deepens the A/scale rings. At the default (2) the
            behavior is bit-identical to before.
        scale_group: Number of consecutive outer-K tiles whose A+B scale
            atoms are fetched by one wide VMEM op per pack and bounced
            through LDS (1 = off). Trades VMEM instructions for LDS ops;
            the preconditions it needs are asserted below.
        b_addr_split: Whether to carry the B-fragment address as a
            loop-invariant per-lane `voffset` plus a wave-uniform
            whole-tile `soffset` (defaults to False). Hoists the per-tile
            address chain out of the unrolled K loop; pays only where the
            warp tile is wide enough to amortise the scalar setup.
        matrix_format: `f8f6f4` operand encoding for A and B. A lane covers
            32 K-elements in every format; the bytes that occupies -- 16
            (FP4), 24 (FP6), 32 (FP8) -- is derived from it.
    """

    # WM is locked to BM — single warp along M for the preb (no-LDS-B) path.
    comptime WM = Self.BM

    comptime a_bits = Self.matrix_format.bits_per_element()
    comptime b_bits = Self.matrix_format.bits_per_element()
    comptime bits_per_element = Self.a_bits

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 128

    comptime num_b_slots = Self.pipeline_depth if Self.b_prefetch else 1
    # A LDS stays double-buffered; the scale + A-register rings deepen to S
    # only on the co-deepened (pipeline_depth>2) path so consumed loads are old.
    comptime num_a_slots = 2 if Self.b_prefetch else 1
    comptime num_scale_slots = Self.pipeline_depth if Self.pipeline_depth > 2 else 1
    comptime num_a_load_slots = Self.pipeline_depth if Self.pipeline_depth > 2 else 1

    comptime MmaOpType = BlockScaledMmaOp_PreB[
        mma_shape=IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K),
        warp_tile=IndexList[3](Self.WM, Self.WN, Self.BK_ELEMS),
        num_b_slots=Self.num_b_slots,
        num_scale_slots=Self.num_scale_slots,
        scale_group=Self.scale_group,
        b_addr_split=Self.b_addr_split,
        matrix_format=Self.matrix_format,
    ]

    comptime num_m_mmas = Self.MmaOpType.num_m_mmas
    comptime num_n_mmas = Self.MmaOpType.num_n_mmas
    comptime num_k_mmas = Self.MmaOpType.num_k_mmas
    comptime MMA_K_BYTES = Self.MmaOpType.MMA_K_BYTES
    comptime c_frag_size = Self.MmaOpType.c_frag_size

    comptime BK_BYTES = (Self.BK_ELEMS * Self.a_bits) // 8
    comptime B_BK_BYTES = (Self.BK_ELEMS * Self.b_bits) // 8

    comptime num_warps_m = 1
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps = Self.num_warps_n
    comptime num_threads = Self.num_warps * WARP_SIZE

    comptime simd_width = simd_width_of[DType.uint8]()

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.num_threads)
        )
    )
    @staticmethod
    def run[
        out_dtype: DType,
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_pre_layout: TensorLayout,
        sfa_layout: TensorLayout,
        sfb_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_pre_engine: TensorEngine,
        sfa_engine: TensorEngine,
        sfb_engine: TensorEngine,
        N: Int,
        K_BYTES: Int,
    ](
        c: TileTensor[out_dtype, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[.uint8, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b_pre: TileTensor[
            .uint8, b_pre_layout, ImmutAnyOrigin, Engine=b_pre_engine
        ],
        sfa: TileTensor[
            .float8_e8m0fnu, sfa_layout, ImmutAnyOrigin, Engine=sfa_engine
        ],
        sfb: TileTensor[
            .float8_e8m0fnu, sfb_layout, ImmutAnyOrigin, Engine=sfb_engine
        ],
        n_tile_idx: Int,
        m_tile_idx: Int,
    ):
        comptime A_K_BYTES = a_layout.static_shape[1]
        comptime assert (
            A_K_BYTES % Self.BK_BYTES == 0
        ), "A's K extent must be a multiple of BK_BYTES"

        comptime assert (
            Self.BM == 16 or Self.BM % 32 == 0
        ), "preshuffled scales require BM == 16 or BM % 32 == 0"
        comptime assert (
            Self.WN == 16 or Self.WN % 32 == 0
        ), "preshuffled scales require WN == 16 or WN % 32 == 0"
        comptime assert Self.num_k_mmas % 2 == 0, (
            "preshuffled scales require num_k_mmas % 2 == 0 (BK_ELEMS % 256"
            " == 0)"
        )
        comptime assert (
            N % 32 == 0
        ), "N must be a multiple of 32 (= 16 * mn_pack) for preshuffled scales"

        comptime num_tiles = A_K_BYTES // Self.BK_BYTES
        comptime assert Self.scale_group in (
            1,
            4,
        ), "scale_group must be 1 (off) or 4 (one dwordx4 per pack)"
        comptime assert Self.scale_group == 1 or (
            Self.b_prefetch
            and Self.pipeline_depth == 2
            and Self.num_k_mmas == 2
            and num_tiles % Self.scale_group == 0
        ), (
            "scale_group > 1 needs the depth-2 prefetch loop, one k_pair per"
            " outer-K tile (BK_ELEMS == 256 at MXFP8), and scale_group"
            " dividing the outer-K tile count"
        )

        comptime K_SCALES = A_K_BYTES // (4 * Self.a_bits)
        # Number of (k_pack=2) scale-dwords needed per outer-K iter.
        comptime mma_k_pair_per_tile = Self.num_k_mmas // 2
        # MN_padded is only used by the layout for shape bookkeeping —
        # the byte-offset math is shape-agnostic (only K_SCALES enters).
        # See address-math notes in block_scaled_preshuffle_layouts.mojo.
        comptime MN_PADDED_PLACEHOLDER = 32

        var M = Int(a.dim[0]())

        var warp_id = warp_id()
        var warp_m, warp_n = divmod(warp_id, Self.num_warps_n)

        # SMEM for A (B comes direct from preshuffled DRAM; so do the
        # scales unless `scale_group > 1` bounces them through LDS).
        # `num_a_slots` buffers laid out slot-major ([slot, BM, BK_BYTES]).
        var a_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[Self.num_a_slots * Self.BM, Self.BK_BYTES]()
        )

        # Per-warp transpose strip for the grouped scale fetch. One atom row
        # per (pack, tile-in-group); zero rows (no LDS at all) when off. The
        # strip scales with num_warps (so with BN/WN) and with scale_packs (so
        # with BM) -- on the enabled BM=128 BN=128 WN=64 band it is 48 rows =
        # 12 KB on top of 64 KB of A, i.e. 76 KB, which still fits two CTAs on
        # a 160 KB CU but with only 8 KB of slack.
        comptime scale_stage_rows = (
            Self.num_warps * Self.MmaOpType.scale_packs * Self.scale_group
        ) if Self.scale_group > 1 else 0
        comptime assert (
            Self.num_a_slots * Self.BM * Self.BK_BYTES
            + scale_stage_rows * Self.MmaOpType.SCALE_ATOM_BYTES
            <= MI355X.shared_memory_per_multiprocessor
        ), (
            "preb LDS budget exceeded; reduce scale_group, num_a_slots or the"
            " tile"
        )
        var scale_smem = stack_allocation[DType.uint8, address_space=.SHARED](
            row_major[scale_stage_rows, Self.MmaOpType.SCALE_ATOM_BYTES]()
        )

        var b_loader = PreshuffledBLoader[
            N=N,
            K_BYTES=K_BYTES,
            cache_policy=Self.b_cache_policy,
            lane_bytes=24 if Self.b_bits == 6 else 16,
        ](b_pre)
        # Bitcast scales' float8_e8m0fnu to uint8 — same byte representation,
        # the PreshuffledScaleLoader expects uint8 buffers.
        var sfa_u8 = TileTensor(sfa.ptr.bitcast[UInt8](), sfa.layout)
        var sfb_u8 = TileTensor(sfb.ptr.bitcast[UInt8](), sfb.layout)
        var a_scale_loader = PreshuffledScaleLoader[
            MN_padded=MN_PADDED_PLACEHOLDER, K_SCALES=K_SCALES
        ](sfa_u8)
        var b_scale_loader = PreshuffledScaleLoader[
            MN_padded=MN_PADDED_PLACEHOLDER, K_SCALES=K_SCALES
        ](sfb_u8)

        comptime load_thread_cols = Self.BK_BYTES // Self.simd_width
        # Cap rows at BM so small BM (e.g. 16) with many warps doesn't ask
        # the load layout for more rows than the A tile has.
        comptime load_thread_rows = _largest_divisor_at_most[Self.BM](
            min(Self.num_threads // load_thread_cols, Self.BM)
        )
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        comptime a_loads_per_tile = Self.BM // load_thread_rows
        comptime load_active_threads = load_thread_rows * load_thread_cols
        comptime a_reg_elems = Self.BM * Self.BK_BYTES // load_active_threads

        # this is of size BM x The entire matrix row
        var a_blockrow = a.tile[Self.BM, A_K_BYTES](m_tile_idx, 0)

        # A DRAM-load landing ring (num_a_load_slots; 1 at the depth-2 default).
        var a_load_reg = stack_allocation[DType.uint8, address_space=.LOCAL](
            row_major[Self.num_a_load_slots, a_reg_elems]()
        )

        var a_loader = RegTileLoader[.uint8, load_layout](
            a_blockrow,
            bounds_from=a,
        )
        # Shared swizzled DRAM->LDS loader (the fp8 4wave/ping-pong path uses
        # the same one). Producer byte-swizzle == the consumer read swizzle.
        # Only consumed on the dram_to_lds path.
        var a_lds_loader = TileLoaderLDS[
            .uint8,
            Self.BM,
            Self.BK_BYTES,
            stride=type_of(a_blockrow).static_stride[0],
            num_loading_warps=Self.num_warps,
            swizzle=a_lds_swizzle[Self.BK_BYTES](),
            load_width=_lds_load_width[
                Self.BK_BYTES, Self.BM, Self.simd_width
            ](),
            use_full_tile_width=True,
        ](a_blockrow, warp_id, Int(lane_id()))

        var warp_m_off_global = m_tile_idx * Self.BM
        var warp_n_off_global = n_tile_idx * Self.BN + warp_n * Self.WN

        var mma_op = Self.MmaOpType(warp_m_off_global, warp_n_off_global)

        var c_writer = RegTileWriter[
            out_dtype, Self.MMA_M, WARP_SIZE // Self.MMA_M
        ](c)

        var k_counter = 0

        @always_inline
        @__parameter
        def load_a_tile_from_dram[reg_slot: Int = 0]():
            # Register-bounce load into landing-ring slot `reg_slot` (no-op in dram_to_lds mode).
            comptime if not Self.dram_to_lds:
                var a_block = a_blockrow.tile[Self.BM, Self.BK_BYTES](
                    0, k_counter
                )
                # Idle threads past the A load layout (only matters when BM is
                # small enough that load_thread_rows is capped at BM, e.g.
                # BM=16 with many warps_n).
                if thread_idx.x < load_active_threads:
                    a_loader.load(
                        a_load_reg.tile[1, a_reg_elems](reg_slot, 0),
                        a_block.vectorize[1, Self.simd_width](),
                    )
                k_counter += 1

        @always_inline
        @__parameter
        def a_smem_slot(
            slot: Int,
        ) -> type_of(a_smem.tile[Self.BM, Self.BK_BYTES](0, 0)):
            return a_smem.tile[Self.BM, Self.BK_BYTES](slot, 0)

        @always_inline
        @__parameter
        def copy_a_tile_to_smem[reg_slot: Int = 0](slot: Int):
            comptime if Self.dram_to_lds:
                # DRAM->LDS via the shared swizzled loader (TileLoaderLDS does
                # the readfirstlane->m0 base + source-side byte swizzle +
                # buffer_load_*_lds internally). k_offset selects the K-tile
                # column; the block's M origin is folded into a_blockrow.
                a_lds_loader.load_tile(
                    a_smem_slot(slot), 0, k_counter * Self.BK_BYTES
                )
                k_counter += 1
                # Drain this wave's DMA so the next barrier publishes a
                # complete LDS tile cross-wave.
                s_waitcnt[vmcnt=0]()
            else:
                # XOR-16 swizzled write (matches load_a_frag_from_smem read).
                # load_layout row_major[rows, cols] maps thread t -> tile
                # (row = t // cols, col_byte = (t % cols) * simd_width); the v
                # loop strides BM by load_thread_rows. Swizzle the flat in-tile
                # byte offset before the raw_store.
                if thread_idx.x < load_active_threads:
                    comptime swizzle = a_lds_swizzle[Self.BK_BYTES]()
                    var a_smem_dst = a_smem_slot(slot)
                    var t = thread_idx.x
                    var base_row = t // load_thread_cols
                    var col_byte = (t % load_thread_cols) * Self.simd_width
                    # `base_row < load_thread_rows`, so a power-of-two row
                    # stride makes the swizzle distribute:
                    #   swizzle(row_off + x) == (swizzle(x) ^ d_v) + row_off
                    # which hoists the lane term out of the unrolled K loop
                    # instead of rematerialising every address per tile.
                    comptime row_stride_pow2 = (
                        load_thread_rows.is_power_of_two()
                    )
                    comptime for v in range(a_loads_per_tile):
                        comptime row_off = (
                            v * load_thread_rows * Self.BK_BYTES
                        )
                        var off: Int
                        comptime if row_stride_pow2:
                            comptime d_v = swizzle(row_off) ^ row_off
                            off = (
                                swizzle(base_row * Self.BK_BYTES + col_byte)
                                ^ d_v
                            ) + row_off
                        else:
                            off = swizzle(
                                base_row * Self.BK_BYTES + col_byte + row_off
                            )
                        a_smem_dst.raw_store[width=Self.simd_width](
                            off,
                            a_load_reg.raw_load[width=Self.simd_width](
                                reg_slot * a_reg_elems + v * Self.simd_width
                            ),
                        )

        @always_inline
        @__parameter
        def load_scales_for_iter[slot: Int = 0](k_pair_base: Int):
            """Issues all A+B preshuffled scale-dword loads for one outer-K iter.

            `k_pair_base = k_iter * mma_k_pair_per_tile` is the absolute
            scale-pack offset; each k_pair advances by 1 (S_K_BLOCK = 8
            K-scales = 2 k_tiles). `slot` selects the scale-ring slot (comptime).
            """
            comptime for k_pair in range(mma_k_pair_per_tile):
                mma_op.load_a_scales_preshuffled[k_pair=k_pair, slot=slot](
                    a_scale_loader,
                    warp_m_off_global,
                    k_pair_base + k_pair,
                )
                mma_op.load_b_scales_preshuffled[k_pair=k_pair, slot=slot](
                    b_scale_loader,
                    warp_n_off_global,
                    k_pair_base + k_pair,
                )

        @always_inline
        @__parameter
        def s_setprio[priority: Int16]():
            # Raise wave priority during MFMA clusters so the matrix unit
            # isn't preempted by memory-issuing waves; lower it for loads.
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        @always_inline
        @__parameter
        def _sched_barrier_zero():
            # Hard reorder fence: pins surrounding instrs to source order so the
            # scheduler can't hoist the interleaved B loads back into one block.
            llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))

        @always_inline
        @__parameter
        def _s_barrier_raw():
            # Bare s_barrier (no vmcnt/lgkmcnt release) so in-flight B DMAs
            # cross it; stdlib barrier() forces vmcnt(0) and kills the prefetch.
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        # Splits the num_k_mmas MFMA chain into mfma_cluster-sized groups so
        # the epilogue can bracket each with s_setprio[1]/[0].
        comptime n_clusters = ceildiv(Self.num_k_mmas, Self.mfma_cluster)

        @always_inline
        @__parameter
        def mma_chain_plain[
            b_slot: Int, a_slot: Int = b_slot, scale_slot: Int = 0
        ]():
            # a_slot/scale_slot default to b_slot/0 (unchanged callers); the
            # co-deepened path passes distinct ring indices.
            var a_warp = a_smem_slot(a_slot).tile[Self.WM, Self.BK_BYTES](
                warp_m, 0
            )
            s_setprio[1]()
            comptime for k in range(Self.num_k_mmas):
                mma_op.load_a_frag_from_smem[k](a_warp)
                mma_op.mma[k, slot=b_slot, scale_slot=scale_slot]()
            s_setprio[0]()

        @always_inline
        @__parameter
        def mma_chain_epilogue[slot: Int]():
            # Last resident tile, no B prefetch left to overlap.
            comptime if Self.cluster_drain_sched:
                # Fully drain the slot's B + scales first (a partial drain
                # needs newer ops behind the consumed slot; the epilogue has
                # none), then keep per-cluster setprio bracketing.
                s_waitcnt[vmcnt=0]()
                var a_warp = a_smem_slot(slot).tile[Self.WM, Self.BK_BYTES](
                    warp_m, 0
                )
                comptime for c in range(n_clusters):
                    comptime k_lo = c * Self.mfma_cluster
                    comptime k_hi = min(
                        k_lo + Self.mfma_cluster, Self.num_k_mmas
                    )
                    s_setprio[1]()
                    comptime for k in range(k_lo, k_hi):
                        mma_op.load_a_frag_from_smem[k](a_warp)
                        mma_op.mma[k, slot=slot]()
                    s_setprio[0]()
            else:
                mma_chain_plain[slot]()

        # TODO use comptime pipeline scheduler

        comptime if (
            Self.b_prefetch
            and Self.pipeline_depth > 2
            and num_tiles >= Self.pipeline_depth
        ):
            # Co-deepened S-deep prefetch: B-frag, scales, and the A DRAM landing
            # are all issued S-1 tiles ahead so the vmcnt wait at consume leaves
            # the newer loads outstanding; the A LDS store stays 1-ahead, published
            # by the non-draining barrier (lgkmcnt + bare s_barrier, no vmcnt drain).
            comptime S = Self.pipeline_depth

            # Prologue: prime tiles [0, S-2] of {A-dram, B-frag, scales}.
            comptime for t in range(S - 1):
                load_a_tile_from_dram[reg_slot=t]()
                comptime for k in range(Self.num_k_mmas):
                    mma_op.load_b_frag_preshuffled[k, slot=t](
                        b_loader, warp_n_off_global, t * Self.BK_BYTES
                    )
                load_scales_for_iter[slot=t](t * mma_k_pair_per_tile)
            copy_a_tile_to_smem[reg_slot=0](0)
            s_waitcnt[lgkmcnt=0]()
            _s_barrier_raw()

            # Steady: MFMA tile i; prefetch tile i+S-1; store tile i+1's A.
            comptime for i in range(num_tiles):
                comptime b_pf = i + S - 1
                comptime if b_pf < num_tiles:
                    load_a_tile_from_dram[reg_slot=b_pf % S]()
                    comptime for k in range(Self.num_k_mmas):
                        mma_op.load_b_frag_preshuffled[k, slot=b_pf % S](
                            b_loader,
                            warp_n_off_global,
                            b_pf * Self.BK_BYTES,
                        )
                    load_scales_for_iter[slot=b_pf % S](
                        b_pf * mma_k_pair_per_tile
                    )

                mma_chain_plain[b_slot=i % S, a_slot=i % 2, scale_slot=i % S]()

                comptime if i + 1 < num_tiles:
                    # Store tile i+1's A (already in a_load_reg) via non-draining publish.
                    copy_a_tile_to_smem[reg_slot=(i + 1) % S]((i + 1) % 2)
                    s_waitcnt[lgkmcnt=0]()
                    _s_barrier_raw()
            barrier()
        elif Self.b_prefetch:
            # Depth-2 outer-K software pipeline (1-deep A prime).
            #
            # Prologue: A (smem), all B fragments (slot 0), and the iter-0
            # scales; a grouped window is staged before A to share its `vmcnt`.
            comptime if Self.scale_group > 1:
                mma_op.stage_scale_group(
                    a_scale_loader,
                    b_scale_loader,
                    warp_m_off_global,
                    warp_n_off_global,
                    0,
                )
            load_a_tile_from_dram()
            copy_a_tile_to_smem(0)
            comptime for k in range(Self.num_k_mmas):
                mma_op.load_b_frag_preshuffled[k, slot=0](
                    b_loader, warp_n_off_global, 0
                )
            comptime if Self.scale_group > 1:
                mma_op.publish_scale_group(scale_smem, warp_id)
                mma_op.read_scale_group[0](scale_smem, warp_id)
            else:
                load_scales_for_iter(0)
            barrier()

            # Steady state: for each i in [0, num_tiles-1), MFMA iter i
            # from `cur_slot` while prefetching iter i+1's B into
            # `nxt_slot`. A SMEM is refilled before the barrier so iter
            # i+1's MFMAs can read it next pass. Scales are reloaded
            # post-barrier from the preshuffled tensor.
            comptime for i in range(num_tiles - 1):
                comptime cur_slot = i % 2
                comptime nxt_slot = (i + 1) % 2
                var nxt_k_byte_base = (i + 1) * Self.B_BK_BYTES

                # Lead the tile so the A landing registers' existing `vmcnt`
                # covers the fetch -- no new drain, B loads stay in flight.
                comptime if (
                    Self.scale_group > 1 and (i + 1) % Self.scale_group == 0
                ):
                    mma_op.stage_scale_group(
                        a_scale_loader,
                        b_scale_loader,
                        warp_m_off_global,
                        warp_n_off_global,
                        (i + 1) * mma_k_pair_per_tile,
                    )

                comptime if Self.cluster_drain_sched:
                    # A goes first: gfx9 has one in-order `vmcnt`, so issuing
                    # it last would force its `ds_write` waits to also retire
                    # the next tile's B loads instead of leaving them in flight.
                    load_a_tile_from_dram()
                    _sched_barrier_zero()

                    # issue next-tile B[k] spread
                    # between the current-tile MFMA phases, each group pinned by
                    # sched_barrier(0) so the scheduler can't re-block the loads
                    # into one burst (the front-load we want to break apart).
                    var a_warp = a_smem_slot(cur_slot).tile[
                        Self.WM, Self.BK_BYTES
                    ](warp_m, 0)
                    comptime for k in range(Self.num_k_mmas):
                        mma_op.load_b_frag_preshuffled[k, slot=nxt_slot](
                            b_loader, warp_n_off_global, nxt_k_byte_base
                        )
                        _sched_barrier_zero()
                        s_setprio[1]()
                        mma_op.load_a_frag_from_smem[k](a_warp)
                        mma_op.mma[k, slot=cur_slot]()
                        s_setprio[0]()
                        _sched_barrier_zero()
                else:
                    comptime for k in range(Self.num_k_mmas):
                        mma_op.load_b_frag_preshuffled[k, slot=nxt_slot](
                            b_loader, warp_n_off_global, nxt_k_byte_base
                        )
                    mma_chain_plain[cur_slot]()
                    load_a_tile_from_dram()

                # Double-buffered A: iter i reads `cur_slot` and writes the
                # next tile into `nxt_slot`, so the old WAR barrier here is
                # gone. The single end-of-iter barrier covers both RAW (write
                # nxt -> read nxt next iter) and WAR (read cur -> overwrite cur
                # next iter, whose nxt == cur).
                copy_a_tile_to_smem(nxt_slot)
                comptime if Self.scale_group > 1:
                    comptime phase = (i + 1) % Self.scale_group
                    comptime if phase == 0:
                        mma_op.publish_scale_group(scale_smem, warp_id)
                    mma_op.read_scale_group[phase](scale_smem, warp_id)
                else:
                    load_scales_for_iter((i + 1) * mma_k_pair_per_tile)
                comptime if Self.cluster_drain_sched:
                    # Publish the A LDS tile cross-wave (lgkmcnt) but let the
                    # next-tile B DMAs keep streaming across the barrier.
                    s_waitcnt[lgkmcnt=0]()
                    _s_barrier_raw()
                elif Self.pipeline_depth > 2:
                    # Non-draining sync so in-flight B DMAs survive the barrier.
                    s_waitcnt[lgkmcnt=0]()
                    _s_barrier_raw()
                else:
                    barrier()

            # Epilogue: MFMA the last iter from its slot.
            comptime last_slot = (num_tiles - 1) % 2
            mma_chain_epilogue[last_slot]()
            barrier()
        else:
            for k_iter in range(num_tiles):
                var a_slot = k_iter % Self.num_a_slots
                load_a_tile_from_dram()
                copy_a_tile_to_smem(a_slot)
                load_scales_for_iter(k_iter * mma_k_pair_per_tile)
                barrier()

                var a_warp = a_smem_slot(a_slot).tile[Self.WM, Self.BK_BYTES](
                    warp_m, 0
                )
                var k_byte_base = k_iter * Self.B_BK_BYTES

                comptime for k in range(Self.num_k_mmas):
                    mma_op.load_a_frag_from_smem[k](a_warp)
                    mma_op.load_b_frag_preshuffled[k](
                        b_loader, warp_n_off_global, k_byte_base
                    )
                    mma_op.mma[k]()
                barrier()

        var c_reg = mma_op.accum_tile()
        var c_block = c.tile[Self.BM, Self.BN](m_tile_idx, n_tile_idx)
        var c_warp = c_block.tile[Self.WM, Self.WN](warp_m, warp_n)

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma in range(Self.num_n_mmas):
                c_writer.store(
                    c_warp.tile[Self.MMA_M, Self.MMA_N](m_mma, n_mma).vectorize[
                        1, Self.c_frag_size
                    ](),
                    c_reg.tile[1, Self.c_frag_size](m_mma, n_mma),
                )
