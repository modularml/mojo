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
"""KV cache buffers for MHA/MLA prefill and decode kernels.

Provides KVCacheIterator (TileTensor-based DRAM tile iteration), the
KVBufferConfig trait + K/V implementors, and three KV buffer structs:

- KVBuffer: double-buffered DMA + LDS + register tile management
  used by MHA/MLA prefill (owns its DRAM iterator).
- DecodeStreamingKVBuffer: single-buffer per-strip DMA used by the
  streaming decode kernel (takes an external DRAM tile per iteration).
- DecodeKVBuffer: double-buffered register staging used by the decode
  mirror path (parametrized by KVBufferConfig for K vs V roles).

TileTensor is used throughout:
  - DRAM tiles: TileTensor with Scalar valid_rows (KVCacheIterator)
  - SMEM sub-tiles: `.tile()` views on a strided parent TileTensor that
    mirrors the blocked (BN × BK) SMEM layout
  - DMA: SubTileLoaderLDS / RegTileLoader (both src and dst are TileTensor)
  - LDS loads: KVMmaOp.load_prefill / load_v_bf16 / load_v_fp8_strip
    (TileTensor SMEM -> reg-tile fragments)
  - MMA register tiles: TileTensor in LOCAL with stack_allocation
"""

from std.math import ceildiv
from std.math.uutils import umod, ufloordiv
from std.sys import simd_width_of, llvm_intrinsic
from max.gpu import WARP_SIZE, lane_id
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    Layout,
    MixedLayout,
    TensorLayout,
    TileTensor,
    DefaultEngine,
)
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation
from layout.tensor_core import TiledTensorCore
from layout.swizzle import Swizzle
from std.utils.numerics import get_accum_type
from nn.attention.mha_operand import MHAOperand
from .mma import KVMmaOp, TiledMmaOp
from .utils import get_warp_coords
from structured_kernels.amd_tile_io import (
    RegTileLoader,
    RegTileWriterLDS,
    SubTileLoaderLDS,
    TiledMmaLoader,
    load_lds_fragment,
)

from std.utils import IndexList


struct KVCacheIterator[
    cache_t: MHAOperand,
    tile_size: Int,
    kv_num_heads: Int,
    depth: Int,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
]:
    """TileTensor-based DRAM tile iterator.

    Returns a TileTensor with Scalar for the row dimension (valid-row
    count) and ComptimeInt for depth and strides. No RuntimeLayout storage.

    When cache_depth != depth, the DRAM stride uses cache_depth (e.g., MLA
    K_rope reads 64 columns from a 576-wide cache row). head_dim_offset
    shifts the column start (e.g., skip to rope portion at column 512).

    Parameters:
        cache_t: MHA operand wrapping the KV cache pointer and dtype.
        tile_size: Number of rows loaded per DRAM tile.
        kv_num_heads: Number of KV heads in the cache.
        depth: Per-head depth read from each cache row (head dimension).
        cache_depth: Row stride of the paged KV cache in elements
            (defaults to `depth`).
        head_dim_offset: Column offset within each cache row, used to
            skip to a sub-region such as the rope portion (defaults to
            `0`).
    """

    comptime GmemTileLayout = MixedLayout[
        Coord[Int64, ComptimeInt[Self.depth]].element_types,
        Coord[
            ComptimeInt[Self.kv_num_heads * Self.cache_depth], ComptimeInt[1]
        ].element_types,
    ]
    comptime GmemTileType = TileTensor[
        Self.cache_t.dtype, Self.GmemTileLayout, ImmutAnyOrigin
    ]

    var cache: Self.cache_t
    var end: Int
    var tile_start_row: Int
    var batch_idx: Int
    var kv_head_idx: Int

    @always_inline
    def __init__(
        out self,
        cache: Self.cache_t,
        batch_idx: Int,
        kv_head_idx: Int,
        end: Int,
    ):
        self.cache = cache
        self.end = end
        self.tile_start_row = 0
        self.batch_idx = batch_idx
        self.kv_head_idx = kv_head_idx

    @always_inline
    def next_tile(mut self) -> Self.GmemTileType:
        """Returns a TileTensor for the next DRAM tile."""
        var valid_rows = max(
            0,
            min(Self.tile_size, self.end - self.tile_start_row),
        )
        var tile = self.cache.block_paged_tile[Self.tile_size](
            UInt32(self.batch_idx),
            UInt32(self.tile_start_row),
            UInt32(self.kv_head_idx),
            Self.GmemTileLayout(
                Coord(
                    Int64(valid_rows),
                    Idx[Self.depth],
                ),
                Coord(
                    Idx[Self.kv_num_heads * Self.cache_depth],
                    Idx[1],
                ),
            ),
            UInt32(Self.head_dim_offset),
        )
        self.tile_start_row += Self.tile_size
        return tile

    @always_inline
    def increment(mut self):
        self.tile_start_row += Self.tile_size


@always_inline
def _get_k_swizzle[mma_m: Int, bk: Int]() -> Optional[Swizzle]:
    """K swizzle for decode.

    XORs upper row bits into lower address bits to spread different rows
    across LDS bank groups in the col_major thread distribution. `bk`
    here is the SMEM physical block width (`_bk_smem`), NOT the MMA strip
    width; the swizzle math operates on per-block addresses.

    Two shapes, by MFMA M-dim:

    - `mma_m == 16` (16x16x{32,64,128}): `Swizzle(3, 0, 3)`. At
      bk=128 (8 vecs/row at simd_w=16B) this XORs vec_idx[3:6]
      (= 8 consecutive rows) into vec_idx[0:3] (= col-in-row).
      At bk=64 (4 vecs/row) the same `S=3` shift XORs row bits
      m[1..3] (skipping m[0]) into vec_idx[0..2], reaching all 8
      distinct bank values across the 16 lanes per col_vec AND
      simultaneously spreading V's `ds_read_tr8_b64` accesses to
      V's structural conflict floor. A shallower `Swizzle(2, 0, 2)`
      is not enough: `S=2` only XORs `m[0..1]`, the LSBs that
      least-distinguish 16 rows, leaving 2-way conflicts behind.
      At bk=32 (BF16 16x16x32 decode) the same swizzle applies.
    - `mma_m == 32` (32x32x{16,64}): `Swizzle(3, 0, 4)` matches the
      32x32 lane geometry (4 16B vecs per 32-row tile).
    """
    comptime if mma_m == 32:
        return Swizzle(3, 0, 4)
    return Swizzle(3, 0, 3)


struct KVBuffer[
    kv_t: MHAOperand,
    //,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
    full_kv: Bool = True,
    cache_depth: Int = depth,
    head_dim_offset: Int = 0,
    reg_chunk_depth: Int = depth,
    reg_chunk_keys: Int = WN,
    smem_depth: Int = depth,
    # SMEM physical block width. When `bk_smem < BK`, the SMEM stride is
    # `bk_smem` and each MMA K=BK strip is composed of `BK / bk_smem`
    # adjacent SMEM blocks. Used by MLA decode at depth=576 (BK=128,
    # bk_smem=64) to avoid wasting an 8 KB pad on a partial block.
    bk_smem: Int = BK,
    # Number of SMEM stages (double-buffer depth). Defaults to 2, so every
    # existing caller is byte-identical. A single-block consumer with no KV
    # streaming loop (e.g. MSA prefill: one `load_from_dram[0]`/
    # `load_from_shared(0)`) never touches the 2nd stage, so it can pass 1 to
    # halve the SMEM allocation.
    num_smem_stages: Int = 2,
]:
    """KV cache buffer managing DMA, LDS staging, and register tiles.

    Handles the full data path: DRAM -> LDS (shared memory) -> registers.

    SMEM is navigated via `.tile()` on a strided parent TileTensor whose
    (BK, BN) strides encode the blocked layout (num_repeats contiguous
    BN×BK blocks per stage, two stages). Stage selection and in-stage
    block selection both happen via the tile column index, no pointer
    arithmetic required. smem_mma_subtile is still used for V-operand
    MMA sub-tiles which have mma_cols != BK.

    When full_kv=True (depth<=256), each SMEM stage holds BN x smem_depth
    elements: the full tile. When full_kv=False (depth=512), each stage
    holds only BN x BK elements, and the caller iterates over BK blocks.

    `smem_depth` defaults to `depth`. It exists for per-warp V buffers
    whose `depth` is smaller than `BK` (e.g. depth_per_warp=16 with
    BK=32): the SMEM layout stays valid with `smem_depth = max(depth,
    BK)` while `depth` keeps driving register-tile sizing and the column
    count read by `load_from_shared`.

    MMA register tiles (mma_tile) are TileTensor in LOCAL address space.
    TiledMmaOp (mma.mojo) handles SMEM→register loads and MMA dispatch.

    Parameters:
        kv_t: MHA operand wrapping the KV cache pointer and dtype
            (inferred).
        mma_shape: MFMA instruction shape `[M, N, K]` used for register
            tiling.
        swizzle: LDS bank-conflict swizzle for SMEM loads and stores, or
            `None`.
        BN: Number of keys per DRAM tile (block tile extent along the
            key axis).
        WN: Number of keys per warp tile (warp tile extent along the key
            axis).
        BK: Strip width along the MMA K dimension of one DRAM-to-SMEM
            load.
        num_threads: Number of threads in the block driving DMA and
            register distribution.
        depth: Per-head depth of the KV cache (head dimension).
        kv_num_heads: Number of KV heads in the cache.
        transpose: `True` for the K operand (column strips from `BN x
            depth`), `False` for V.
        full_kv: When `True`, each SMEM stage holds the full `BN x
            smem_depth` tile; when `False`, each stage holds only `BN x
            BK` and the caller iterates over BK blocks (defaults to
            `True`).
        cache_depth: Row stride of the paged KV cache in elements
            (defaults to `depth`).
        head_dim_offset: Column offset within each cache row, used to
            skip to a sub-region such as the rope portion (defaults to
            `0`).
        reg_chunk_depth: Cap on the K (transpose) register tile depth
            coverage, in elements (defaults to `depth`).
        reg_chunk_keys: Cap on the V (non-transpose) register tile key
            coverage, in keys (defaults to `WN`).
        smem_depth: Depth extent of one SMEM stage; larger than `depth`
            for per-warp V buffers whose `depth` is smaller than `BK`
            (defaults to `depth`).
        bk_smem: SMEM physical block width; when less than `BK`, each MMA
            K strip is composed of `BK / bk_smem` adjacent SMEM blocks
            (defaults to `BK`).
        num_smem_stages: Number of SMEM stages (double-buffer depth); a
            single-block consumer with no streaming loop may pass `1` to
            halve the SMEM allocation (defaults to `2`).
    """

    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth, Self.MMA_N
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K)
    # B-operand fragment size: `num_matrix_reg[MMA_K, MMA_N]`.
    # For BF16 [32,32,16]: 8. For FP8 [32,32,64]: 32.
    comptime input_frag_size = (Self.MMA_K * Self.MMA_N) // WARP_SIZE
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime num_k_tiles = ceildiv(
        Self.depth if Self.transpose else Self.WN, Self.BK
    )
    # Register-side strip count. The caller can cap the reg tile so it stays
    # small even when SMEM holds the full tile: K (transpose=True) via
    # reg_chunk_depth (depth coverage), V (transpose=False) via reg_chunk_keys
    # (key coverage). The global strip index maps into the (aliased) reg tile
    # via `bk_tile % _reg_num_k_tiles`; the caller must then drive the strip
    # loop explicitly (load[strip] -> consume) so slots don't clobber early.
    comptime _reg_num_k_tiles = (
        ceildiv(Self.reg_chunk_depth, Self.BK) if Self.transpose else ceildiv(
            Self.reg_chunk_keys, Self.BK
        )
    )

    comptime warp_tile_rows = 32
    comptime num_repeats = Self.smem_depth // Self.bk_smem
    comptime smem_cols = Self.smem_depth if Self.full_kv else Self.bk_smem
    comptime smem_stage_size = Self.BN * Self.smem_cols

    # Strided parent view over the full 2-stage SMEM allocation.
    # Shape (BN, _smem_total_cols) with stride (bk_smem, BN) so that
    # `.tile[tile_rows, bk_smem]((tile_row, tile_col))` produces
    # `tile_row * tile_rows * bk_smem + tile_col * BN * bk_smem`,
    # matching the block-aligned offsets of the blocked
    # (BN × bk_smem) SMEM layout. tile_col indexes linearly over all
    # blocks across both stages, so stage selection happens via
    # coordinate arithmetic rather than pointer arithmetic
    # (col = buffer_idx * blocks_per_stage + block).
    comptime _blocks_per_stage = Self.num_repeats if Self.full_kv else 1
    comptime _smem_total_cols = (
        Self.num_smem_stages * Self._blocks_per_stage * Self.bk_smem
    )
    comptime _SmemParentLayout = MixedLayout[
        Coord[
            ComptimeInt[Self.BN], ComptimeInt[Self._smem_total_cols]
        ].element_types,
        Coord[ComptimeInt[Self.bk_smem], ComptimeInt[Self.BN]].element_types,
    ]
    # Strides for sub-tiles of width bk_smem: plain row-major (bk_smem, 1)
    # so element indexing within a block behaves normally.
    comptime _SmemTileStrides = MixedLayout[
        Coord[ComptimeInt[Self.bk_smem], ComptimeInt[1]].element_types,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]

    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _dma_col_groups = (
        Self.smem_depth // Self.bk_smem
    ) if Self.full_kv else 1
    comptime _total_tiles = (
        Self.BN // Self.warp_tile_rows
    ) * Self._dma_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_tiles, Self._num_warps)
    # A nonuniform tile split has no single safe per-wave outstanding count;
    # zero makes pipelined consumers wait for every wave's DMA to finish.
    comptime vm_instrs_per_load = UInt32(
        Self._tiles_per_warp * 2 if Self._total_tiles % Self._num_warps
        == 0 else 0
    )

    comptime _mma_total_rows = Self.num_mmas * Self.num_k_mmas2 * Self._reg_num_k_tiles
    comptime mma_layout = row_major[
        Self._mma_total_rows, Self.input_frag_size
    ]()
    comptime MMATileType = TileTensor[
        Self.kv_t.dtype,
        type_of(Self.mma_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    comptime KVMmaOpType = KVMmaOp[
        Self.kv_t.dtype,
        Self.mma_shape,
        Self.num_mmas,
        Self.num_k_mmas2,
        Self._reg_num_k_tiles,
        Self.BN,
        Self.BK,
        transpose_b=Self.transpose,
        swizzle=Self.swizzle,
    ]
    var kv_mma_op: Self.KVMmaOpType

    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime SmemParentType = TileTensor[
        Self.kv_t.dtype,
        Self._SmemParentLayout,
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var smem_tile: Self.SmemParentType

    var kv_cache_iter: KVCacheIterator[
        Self.kv_t,
        Self.BN,
        Self.kv_num_heads,
        Self.depth,
        Self.cache_depth,
        Self.head_dim_offset,
    ]

    var warp_id: UInt32

    @always_inline
    def _smem_view(self) -> Self.SmemParentType:
        """Full 2-stage SMEM view with strides (BK, BN)."""
        return self.smem_tile

    @always_inline
    def smem_block_tile[
        tile_rows: Int,
    ](self, tile_row: Int, block_col: Int) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[tile_rows, Self.bk_smem]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ]:
        """Get a (tile_rows, bk_smem) row-major sub-tile from SMEM.

        tile_row indexes along BN (rows within a BN×bk_smem block),
        block_col indexes linearly across all BN×bk_smem blocks in both
        stages.

        Parameters:
            tile_rows: Number of rows in the returned sub-tile.

        Args:
            tile_row: Row index along `BN` within a `BN x bk_smem` block.
            block_col: Linear index across all `BN x bk_smem` blocks in
                both stages.
        """
        return self._smem_view().tile[
            tile_rows,
            Self.bk_smem,
            stride_layout=Self._SmemTileStrides,
        ](Coord(tile_row, block_col))

    @always_inline
    def __init__(
        out self,
        k_cache: Self.kv_t,
        batch_idx: Int,
        head_idx: Int,
        smem_tile: Self.SmemParentType,
        end: Int,
        warp_id: UInt32,
    ):
        self.kv_mma_op = Self.KVMmaOpType()
        self.smem_tile = smem_tile

        self.kv_cache_iter = type_of(self.kv_cache_iter)(
            k_cache, batch_idx, head_idx, end
        )

        self.warp_id = warp_id

    @always_inline
    def load_from_dram[buffer_idx: Int](mut self):
        var gmem_tile = self.kv_cache_iter.next_tile()
        var loader = SubTileLoaderLDS[Self.kv_t.dtype, Self.swizzle](gmem_tile)

        # Stage offset for this buffer_idx within the linear block index.
        comptime _stage_block_base = buffer_idx * Self._blocks_per_stage

        comptime if not Self.full_kv:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime tiles_per_warp = ceildiv(num_row_groups, num_warps)

            comptime for t in range(tiles_per_warp):
                comptime tile_idx = Int(t) * num_warps
                var warp_tile_idx = UInt32(tile_idx) + self.warp_id
                var warp_row = Int(warp_tile_idx)
                var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                    warp_row, _stage_block_base
                )
                var gmem_warp_tile = gmem_tile.tile[
                    Self.warp_tile_rows, Self.bk_smem
                ](warp_row, 0)
                # Default `hoist_scalar_offset=False` keeps the legacy
                # per-iter `Int(src_partitions.ptr) - dram_base` codegen.
                loader.load(smem_warp, gmem_warp_tile)
        elif (
            Self.smem_depth == 64
            and Self.BN <= Self.warp_tile_rows * 2
            and Self.smem_depth // Self.bk_smem >= 2
        ):
            var warp_r, warp_c = divmod(Int(self.warp_id), 2)
            var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                warp_r, _stage_block_base + warp_c
            )
            var gmem_warp_tile = gmem_tile.tile[
                Self.warp_tile_rows, Self.bk_smem
            ](warp_r, warp_c)
            loader.load(smem_warp, gmem_warp_tile)
        else:
            comptime num_warps = Self.num_threads // WARP_SIZE
            comptime num_row_groups = Self.BN // Self.warp_tile_rows
            comptime num_col_groups = Self.smem_depth // Self.bk_smem
            comptime total_tiles = num_row_groups * num_col_groups
            comptime tiles_per_warp = ceildiv(total_tiles, num_warps)
            # Coverage: every tile must be reachable by some (warp, t) pair.
            comptime assert tiles_per_warp * num_warps >= total_tiles

            # Bounds guard only needed when `num_warps` doesn't divide
            # `total_tiles` (e.g. W in {5,7,8} at depth=576/BN=128 → 36 tiles):
            # the last warp's final tile would index past `total_tiles` and
            # over-read. When it divides exactly (e.g. W=4) the unguarded loop
            # below is exact — byte-identical at S=1.
            comptime _needs_dma_guard = (total_tiles % num_warps) != 0
            comptime if _needs_dma_guard:
                comptime for t in range(tiles_per_warp):
                    comptime tile_idx = Int(t) * num_warps
                    var warp_tile = UInt32(tile_idx) + self.warp_id
                    # Mirrors the guarded streaming twin
                    # (`if warp_tile < total_dma_tiles` below).
                    if warp_tile < UInt32(total_tiles):
                        var warp_row, warp_col = divmod(
                            warp_tile, UInt32(num_col_groups)
                        )
                        var smem_warp = self.smem_block_tile[
                            Self.warp_tile_rows
                        ](Int(warp_row), _stage_block_base + Int(warp_col))
                        var gmem_warp_tile = gmem_tile.tile[
                            Self.warp_tile_rows, Self.bk_smem
                        ](Int(warp_row), Int(warp_col))
                        loader.load(smem_warp, gmem_warp_tile)
            else:
                comptime for t in range(tiles_per_warp):
                    comptime tile_idx = Int(t) * num_warps
                    var warp_tile = UInt32(tile_idx) + self.warp_id
                    var warp_row, warp_col = divmod(
                        warp_tile, UInt32(num_col_groups)
                    )
                    var smem_warp = self.smem_block_tile[Self.warp_tile_rows](
                        Int(warp_row), _stage_block_base + Int(warp_col)
                    )
                    var gmem_warp_tile = gmem_tile.tile[
                        Self.warp_tile_rows, Self.bk_smem
                    ](Int(warp_row), Int(warp_col))
                    loader.load(smem_warp, gmem_warp_tile)

        # K-tail padding is handled register-side in `zero_partial_tile_pad`
        # (as the reference does — see that method for the rationale). The
        # SMEM tail bytes for `cols [depth, smem_depth)` of the last K-tile
        # remain at whatever the DMA's OOB-clamp / row-aliasing produced; the K
        # MFMA never reads from those bytes because the per-lane fragment's
        # upper half is overridden with zero after the LDS load.

    # split[N]()[idx] → tile[rows_per_split, cols](idx, 0)
    comptime _rows_per_k_tile = Self.num_mmas * Self.num_k_mmas2
    comptime _rows_per_k_mma = Self.num_mmas

    @always_inline
    def get_mma_tile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self._rows_per_k_mma, Self.input_frag_size]()),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]:
        comptime reg_slot = bk_tile_idx % Self._reg_num_k_tiles
        return self.kv_mma_op.mma_tile_at[reg_slot, k_mma_tile_idx]()

    @always_inline
    def mma_subtile[
        k_mma_tile_idx: Int,
        bk_tile_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self._rows_per_k_mma, Self.input_frag_size]()),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]:
        """Alias for get_mma_tile, kept for decode-call-site symmetry.

        Parameters:
            k_mma_tile_idx: Index of the MFMA tile along the K dimension
                within one BK strip.
            bk_tile_idx: Index of the BK strip along the depth (transpose)
                or key (non-transpose) axis.
        """
        return self.get_mma_tile[k_mma_tile_idx, bk_tile_idx]()

    @always_inline
    def zero_partial_tile_pad(self):
        """Register-side zero for the OOB tail of the partial K-tile.

        When `depth % BK != 0`, the last K-tile (i = depth // BK) spans
        `BK` K-positions but only `valid_cols = depth - i*BK` are valid;
        the trailing `BK - valid_cols` are pad and must read as zero.

        Per-lane K-fragment layout: the `input_frag_size` elements per
        lane interleave across MFMA-K such that the LOWER
        `valid_per_lane = input_frag_size * valid_cols / BK` elements
        correspond to the valid K-range and the UPPER
        `input_frag_size - valid_per_lane` elements correspond to the
        pad. Zero the upper portion per lane.

        The reference pre-zeros half of the partial-tile K-fragment dwords
        once and reuses; we re-zero each K LDS load because the LDS loader
        fills the whole reg tile.

        A no-op when `depth % BK == 0`.

        NOTE: keep in sync with `QRegisterBuffer.__init__`'s partial-tile
        zero in `buffers.mojo`. Both sites share the upper-half-is-pad
        assumption (asserted below); a future config that violates it
        (`valid_cols > BK/2`) needs a different zero pattern in both.
        """
        comptime if Self.depth % Self.BK != 0:
            comptime partial_bk_tile = Self.depth // Self.BK
            comptime valid_cols_in_partial = (
                Self.depth - partial_bk_tile * Self.BK
            )
            # Upper-half-is-pad invariant: today (depth=576, BK=128)
            # `valid_cols` is exactly BK/2, so the lo half of each
            # lane's per-tile fragment covers the valid K-range and the
            # hi half is pad. A future config with `valid_cols > BK/2`
            # would split the valid range across both halves and need a
            # different zero pattern.
            comptime assert valid_cols_in_partial <= Self.BK // 2, (
                "zero_partial_tile_pad assumes valid cols fit in the"
                " lower-K half of the per-lane fragment"
            )
            comptime valid_per_lane = (
                Self.input_frag_size * valid_cols_in_partial // Self.BK
            )
            comptime zero_per_lane = Self.input_frag_size - valid_per_lane
            comptime assert (
                zero_per_lane > 0
            ), "zero_partial_tile_pad: zero_per_lane must be positive"
            comptime for k_mma in range(Self.num_k_mmas2):
                # Zero cols [valid_per_lane .. input_frag_size) of each
                # lane's partial-bk-tile fragment — the K-pad portion
                # produced by `load_b`'s `lo.join(hi)` upper half.
                _ = (
                    self.mma_subtile[k_mma, partial_bk_tile]()
                    .tile[Self._rows_per_k_mma, zero_per_lane](0, 1)
                    .fill(0)
                )

    @always_inline
    def load_from_shared(self, buffer: Int):
        # The no-index form loads every strip into the reg tile. When the
        # reg tile is chunked (_reg_num_k_tiles < num_k_tiles) slots alias,
        # so the caller must drive the chunk loop explicitly via the
        # indexed overload.
        comptime assert Self._reg_num_k_tiles == Self.num_k_tiles, (
            "load_from_shared(buffer) requires full-depth reg tile; use"
            " load_from_shared[bk_tile](buffer) for chunked buffers"
        )
        comptime if (
            not Self.transpose
            and Self.kv_t.dtype.is_float8()
            and Self.mma_shape[0] == 16
        ):
            # FP8 16x16x128 V load: paired-lane `ds_read_tr8_b64`.
            # Each (bk_tile, dt) iteration covers one MFMA tile of V
            # (16 depths * 128 keys = 2048 FP8 = 64 lanes * 32 FP8/lane)
            # via 4 `ds_read_tr8_b64` calls at key_base ∈ {0,8,16,24}.
            # Lane partition for the 64-lane wave:
            #   key_group g  = lid // 16     (0..3 -> 16-lane "rows")
            #   pair_idx  p  = (lid%16) // 2 (0..7 -> pair within row;
            #                                 even/odd share same pair)
            #   is_odd    o  = lid % 2       (0 or 1)
            # See `TiledMmaLoader.load_v_fp8_strip_16` for the address
            # arithmetic; per-lane output: V[key=g*32..g*32+31,
            # depth=butterfly(lid%16) + dt*16] where butterfly is the
            # natural depth permutation of `ds_read_tr8_b64`'s two
            # interleaved 8x8 transposes. The MFMA A-operand consumes
            # this permuted layout directly (the 16x16x128 m_h lane
            # mapping IS the butterfly).
            comptime MMA_M_ = Self.mma_shape[0]
            comptime num_depth_tiles = Self.depth // MMA_M_

            var v_base = self.smem_tile.tile[Self.BN, Self.bk_smem](
                0, buffer * Self._blocks_per_stage
            ).ptr

            var lid = lane_id()
            var key_group = Int(ufloordiv(lid, 16))
            var pair_idx = Int(ufloordiv(umod(lid, 16), 2))
            var is_odd = Int(umod(lid, 2))

            var reg_vec = self.kv_mma_op.reg_tile.vectorize[
                1, Self.input_frag_size
            ]()

            comptime for bk_tile in range(Self.num_k_tiles):
                comptime for dt in range(num_depth_tiles):
                    var joined = TiledMmaLoader[
                        Self.kv_t.dtype,
                        Self.mma_shape,
                        swizzle=Self.swizzle,
                    ].load_v_fp8_strip_16[
                        Self.BN, Self.bk_smem, bk_tile, Int(dt)
                    ](
                        v_base, key_group, pair_idx, is_odd
                    )
                    reg_vec[bk_tile * num_depth_tiles + dt, 0] = rebind[
                        type_of(reg_vec[0, 0])
                    ](joined)
        elif not Self.transpose and Self.kv_t.dtype.is_float8():
            # FP8 V vector load using ds_read_tr8_b64 with paired-lane
            # addressing. Replaces ~128 scalar LDS reads with ~16 vector
            # reads (8x fewer instructions).
            #
            # Paired lanes (even/odd) access the same key at depth offsets
            # differing by 8. After the hardware 8x8 transpose, each lane
            # holds 8 different keys at ONE depth (NOT 8 contiguous depths
            # per lane — that's the pre-transpose source layout). The 16
            # lanes in a row collectively cover 16 unique depths; the
            # depth-per-lane mapping is `depth_in_block = lane_in_row +
            # (row_in_warp % 2) * 16`. Two rows within hw0 (depth_base 0
            # and 16) give 32 depths; hw1 shifts keys by +4 for the
            # complementary MFMA C-output column pattern, covering all
            # 64 BN keys per MFMA tile. The per-strip load lives in
            # `TiledMmaLoader.load_v_fp8_strip` — see that method for
            # the addressing details; here we precompute the lane-only
            # coords once and iterate (bk_tile, dt).
            var v_base = self.smem_tile.tile[Self.BN, Self.BK](
                0, buffer * Self._blocks_per_stage
            ).ptr

            # Per-lane address components (computed once, reused across
            # all bk_tile iterations by `KVMmaOp.load_v_fp8_strip`).
            var lid = lane_id()
            var lane_in_row = umod(lid, 16)
            var pair_idx = ufloordiv(lane_in_row, 2)
            var is_odd = umod(lane_in_row, 2)
            var row_in_warp = ufloordiv(lid, 16)
            var is_hw1 = ufloordiv(lid, 32)
            var rel_key = Int(umod(pair_idx, 4) + ufloordiv(pair_idx, 4) * 8)
            var depth_base = Int(umod(row_in_warp, 2)) * 16 + Int(is_odd) * 8
            var hw_key_shift = Int(is_hw1) * 4

            comptime for bk_tile in range(Self.num_k_tiles):
                self.kv_mma_op.load_v_fp8_strip[bk_tile](
                    v_base, rel_key, hw_key_shift, depth_base
                )
        else:
            comptime for bk_tile in range(Self.num_k_tiles):
                self.load_from_shared[bk_tile](buffer)

    @always_inline
    def load_from_shared[bk_tile: Int](self, buffer: Int):
        var smem_base = self.smem_tile.tile[Self.BN, Self.BK](
            0, buffer * Self._blocks_per_stage
        ).ptr

        comptime if Self.transpose:
            # K (transpose) path: delegate to KVMmaOp.load_prefill.
            # BF16: single load per MMA tile.
            # FP8:  two half-K loads joined (num_packs=2 branch inside
            #       TiledMmaLoader.load_b matches the FP8 MMA K=128 layout).
            comptime num_warps_n = Self.BN // Self.WN
            var warp_col = umod(Int(self.warp_id), num_warps_n)

            comptime if Self.bk_smem < Self.BK:
                # bk_smem-split path: each MMA K=BK strip is composed of
                # `BK / bk_smem` adjacent BN×bk_smem SMEM blocks. For the
                # FP8 16x16x128 case (BK=128, bk_smem=64), that's two
                # blocks per strip. The final strip may be partial when
                # depth % BK != 0 — no hi block, register-zero the upper
                # half of the MMA fragment.
                # The K-split path uses `load_prefill_split` (two 64-wide
                # blocks per 128-wide MMA strip). Generalizing to other
                # ratios would need a `BK / bk_smem`-arity load helper;
                # not implemented today since the only shipping config
                # is depth=576, BK=128, bk_smem=64.
                comptime assert (
                    Self.bk_smem * 2 == Self.BK
                ), "Only bk_smem == BK/2 supported in the K split path"
                comptime num_full_strips = Self.depth // Self.BK
                comptime has_hi = bk_tile < num_full_strips

                var base_block = buffer * Self._blocks_per_stage + bk_tile * 2
                var warp_lo = self.smem_block_tile[Self.wtile_dim0](
                    Int(warp_col), base_block
                )
                # When has_hi=False we still pass warp_lo as the hi arg
                # (it's ignored by load_prefill_split via comptime branch).
                var warp_hi = self.smem_block_tile[Self.wtile_dim0](
                    Int(warp_col),
                    base_block + 1 if has_hi else base_block,
                )

                comptime reg_slot = bk_tile % Self._reg_num_k_tiles
                self.kv_mma_op.load_prefill_split[reg_slot, has_hi](
                    warp_lo, warp_hi
                )
            else:
                var warp_smem = self.smem_block_tile[Self.wtile_dim0](
                    Int(warp_col), buffer * Self._blocks_per_stage + bk_tile
                )

                comptime reg_slot = bk_tile % Self._reg_num_k_tiles
                self.kv_mma_op.load_prefill[reg_slot](warp_smem)

        else:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 V per-strip path — delegate to
                # `KVMmaOp.load_v_fp8_strip`. Single-bk_tile entry point
                # recomputes the lane-only coords each call (the bulk
                # `load_all_from_shared` variant hoists them across a
                # multi-bk loop).
                var lid = lane_id()
                var lane_in_row = umod(lid, 16)
                var pair_idx = ufloordiv(lane_in_row, 2)
                var is_odd = umod(lane_in_row, 2)
                var row_in_warp = ufloordiv(lid, 16)
                var is_hw1 = ufloordiv(lid, 32)
                var rel_key = Int(
                    umod(pair_idx, 4) + ufloordiv(pair_idx, 4) * 8
                )
                var depth_base = (
                    Int(umod(row_in_warp, 2)) * 16 + Int(is_odd) * 8
                )
                var hw_key_shift = Int(is_hw1) * 4

                self.kv_mma_op.load_v_fp8_strip[bk_tile](
                    smem_base, rel_key, hw_key_shift, depth_base
                )
            else:
                # BF16 V path — delegate to `KVMmaOp.load_v_bf16`.
                self.kv_mma_op.load_v_bf16[bk_tile](smem_base)


struct DecodeStreamingKVBuffer[
    kv_t: MHAOperand,
    //,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    num_threads: Int,
    depth: Int,
    kv_num_heads: Int,
    transpose: Bool,
]:
    """Streaming-decode KV buffer: single-buffer SMEM staging with per-strip DMA.

    Unlike KVBuffer, this takes an external DRAM tile per
    outer-loop iteration and loads BK-wide strips one at a time.

    K (transpose=True): BN x BK SMEM, column strips from BN x depth.
    V (transpose=False): BK x depth SMEM (blocked BK x BK), row strips.

    Parameters:
        kv_t: MHA operand wrapping the KV cache pointer and dtype
            (inferred).
        mma_shape: MFMA instruction shape `[M, N, K]` used for register
            tiling.
        swizzle: LDS bank-conflict swizzle for SMEM loads and stores, or
            `None`.
        BN: Number of keys per DRAM tile (block tile extent along the
            key axis).
        WN: Number of keys per warp tile (warp tile extent along the key
            axis).
        BK: Strip width along the MMA K dimension of one per-iteration
            DRAM-to-SMEM load.
        num_threads: Number of threads in the block driving DMA and
            register distribution.
        depth: Per-head depth of the KV cache (head dimension).
        kv_num_heads: Number of KV heads in the cache.
        transpose: `True` for the K operand (column strips from `BN x
            depth`), `False` for V (row strips).
    """

    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime simd_width = simd_width_of[Self.kv_t.dtype]()
    comptime input_frag_size = (Self.MMA_K * Self.MMA_N) // WARP_SIZE

    # MMA tiling within one strip.
    # K (transpose=True): WN MMA tiles per warp (warp tiling in key dim).
    # V (transpose=False): warp's depth portion — each warp handles
    # depth // num_warps_n columns, matching the output register tiling.
    comptime _num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(
        Self.WN if Self.transpose else Self.depth // Self._num_warps_n,
        Self.MMA_N,
    )
    comptime num_k_mmas2 = ceildiv(Self.BK, Self.MMA_K)

    # Register tile: single strip only.
    comptime _mma_total_rows = Self.num_mmas * Self.num_k_mmas2
    comptime mma_layout = row_major[
        Self._mma_total_rows, Self.input_frag_size
    ]()
    comptime MMATileType = TileTensor[
        Self.kv_t.dtype,
        type_of(Self.mma_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    comptime KVMmaOpType = KVMmaOp[
        Self.kv_t.dtype,
        Self.mma_shape,
        Self.num_mmas,
        Self.num_k_mmas2,
        1,  # num_k_tiles: single strip
        Self.BN,
        Self.BK,
        transpose_b=Self.transpose,
        swizzle=Self.swizzle,
    ]

    # DMA tiling: warp-cooperative load of one strip.
    comptime warp_tile_rows = 32
    comptime _num_warps = Self.num_threads // WARP_SIZE
    comptime _strip_rows = Self.BN if Self.transpose else Self.BK
    comptime _strip_cols = Self.BK if Self.transpose else Self.depth
    comptime _num_row_groups = Self._strip_rows // Self.warp_tile_rows
    comptime _num_col_groups = Self._strip_cols // Self.BK
    comptime _total_dma_tiles = Self._num_row_groups * Self._num_col_groups
    comptime _tiles_per_warp = ceildiv(Self._total_dma_tiles, Self._num_warps)

    # Strided parent view over the single-stage K SMEM (BN × BK).
    # Strides (BK, BN) so `.tile[tile_rows, BK]((tile_row, 0))` yields
    # `tile_row * tile_rows * BK`, matching smem_subtile for 1 block.
    # Only used for the K (transpose=True) path; V uses a plain
    # row_major[BK, depth] view constructed inline.
    comptime _KSmemParentLayout = MixedLayout[
        Coord[ComptimeInt[Self.BN], ComptimeInt[Self.BK]].element_types,
        Coord[ComptimeInt[Self.BK], ComptimeInt[Self.BN]].element_types,
    ]
    comptime _KSmemTileStrides = MixedLayout[
        Coord[ComptimeInt[Self.BK], ComptimeInt[1]].element_types,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]

    var kv_mma_op: Self.KVMmaOpType

    @__allow_legacy_any_origin_fields
    var smem_ptr: UnsafePointer[
        Scalar[Self.kv_t.dtype], MutAnyOrigin, address_space=.SHARED
    ]
    var warp_id: UInt32

    @always_inline
    def __init__(
        out self,
        cache: Self.kv_t,
        batch_idx: Int,
        head_idx: Int,
        smem_ptr: UnsafePointer[
            Scalar[Self.kv_t.dtype], MutAnyOrigin, address_space=.SHARED
        ],
        num_keys: Int,
        warp_id: UInt32,
    ):
        self.kv_mma_op = Self.KVMmaOpType()
        self.smem_ptr = smem_ptr
        self.warp_id = warp_id

    @always_inline
    def _k_smem_view(
        self,
    ) -> TileTensor[
        Self.kv_t.dtype,
        Self._KSmemParentLayout,
        MutAnyOrigin,
        address_space=.SHARED,
    ]:
        """Single-stage K SMEM view (BN × BK) with strides (BK, BN)."""
        return TileTensor[
            Self.kv_t.dtype,
            Self._KSmemParentLayout,
            MutAnyOrigin,
            address_space=.SHARED,
        ](self.smem_ptr, Self._KSmemParentLayout())

    @always_inline
    def k_smem_block_tile[
        tile_rows: Int,
    ](self, tile_row: Int) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[tile_rows, Self.BK]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ]:
        """Get a (tile_rows, BK) row-major sub-tile from the K SMEM.

        Single-stage K SMEM has one BN×BK block, so only the row index
        along BN varies.

        Parameters:
            tile_rows: Number of rows in the sub-tile.

        Args:
            tile_row: Row index along `BN` within the `BN x BK` SMEM
                block.
        """
        return self._k_smem_view().tile[
            tile_rows,
            Self.BK,
            stride_layout=Self._KSmemTileStrides,
        ](Coord(tile_row, Idx[0]))

    @always_inline
    def load_from_dram[
        strip_idx: Int
    ](
        self,
        gmem_tile: TileTensor[Self.kv_t.dtype, Engine=DefaultEngine[], ...],
    ):
        """Load one BK-wide strip from an external DRAM tile to SMEM.

        K (transpose=True): columns [strip*BK, (strip+1)*BK] from BN x depth.
        V (transpose=False): rows [strip*BK, (strip+1)*BK] from BN x depth.

        Parameters:
            strip_idx: Strip index along the K dimension (transpose) or
                the row axis (non-transpose) of the DRAM tile.

        Args:
            gmem_tile: External DRAM tile to load the strip from.
        """
        comptime if Self.transpose:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 K decode: BK=32 is narrower than the SubTileLoaderLDS
                # DMA sub-tile (simd_width_of[fp8]()*4 = 64), so use a
                # DRAM→regs→SMEM path (same pattern as V below).
                comptime _k_sw = simd_width_of[Self.kv_t.dtype]()
                comptime _k_btile0 = Self.BN
                comptime _k_btile1 = Self.BK
                comptime _k_thr_rows = (
                    min(
                        Self.num_threads,
                        (_k_btile0 * _k_btile1) // _k_sw,
                    )
                    * _k_sw
                    // _k_btile1
                )
                comptime _k_thr_cols = _k_btile1 // _k_sw
                comptime _k_rows_per_stage = (
                    _k_btile0 * _k_btile1
                ) // Self.num_threads // _k_sw

                var reg_loader = RegTileLoader[
                    Self.kv_t.dtype,
                    row_major[_k_thr_rows, _k_thr_cols](),
                    Self.num_threads,
                ](gmem_tile)

                var load_buf = stack_allocation[
                    Self.kv_t.dtype, address_space=.LOCAL
                ](row_major[_k_rows_per_stage, _k_sw]())

                var dram_strip = gmem_tile.tile[Self.BN, Self.BK](0, strip_idx)
                reg_loader.load(
                    load_buf,
                    dram_strip.vectorize[1, _k_sw](),
                )

                comptime _k_smem_layout = row_major[Self.BN, Self.BK]()
                var k_smem = TileTensor[
                    Self.kv_t.dtype,
                    type_of(_k_smem_layout),
                    MutAnyOrigin,
                    address_space=.SHARED,
                ](self.smem_ptr, _k_smem_layout)

                RegTileWriterLDS[
                    row_major[_k_thr_rows, _k_thr_cols](),
                    Self.swizzle,
                    Self.num_threads,
                ].copy(
                    k_smem.vectorize[1, _k_sw](),
                    load_buf.vectorize[1, _k_sw](),
                )
            else:
                # BF16 K: column strip → BN x BK to SMEM via LDS DMA.
                var loader = SubTileLoaderLDS[Self.kv_t.dtype, Self.swizzle](
                    gmem_tile
                )
                comptime for t in range(Self._tiles_per_warp):
                    comptime tile_base = Int(t) * Self._num_warps
                    var warp_tile = UInt32(tile_base) + self.warp_id
                    if warp_tile < UInt32(Self._total_dma_tiles):
                        var warp_row = Int(warp_tile)
                        var smem_warp = self.k_smem_block_tile[
                            Self.warp_tile_rows
                        ](warp_row)
                        var gmem_warp = gmem_tile.tile[Self.BN, Self.BK](
                            0, strip_idx
                        ).tile[Self.warp_tile_rows, Self.BK](warp_row, 0)
                        loader.load(smem_warp, gmem_warp)
        else:
            # V: row strip → BK x depth to flat SMEM [BK, depth].
            # DRAM→regs→SMEM matching the decode DecodeKVBuffer token_gen V.
            # Flat layout (stride=depth) is required so that load_b
            # sees the correct strides for warp sub-tile distribution.
            comptime _v_sw = simd_width_of[Self.kv_t.dtype]()
            comptime _v_btile0 = Self.BK
            comptime _v_btile1 = Self.depth
            comptime _v_thr_rows = (
                min(
                    Self.num_threads,
                    (_v_btile0 * _v_btile1) // _v_sw,
                )
                * _v_sw
                // _v_btile1
            )
            comptime _v_thr_cols = _v_btile1 // _v_sw
            comptime _v_warp_depth = Self.depth // Self._num_warps_n
            comptime _v_num_mmas_v = ceildiv(_v_warp_depth, Self.MMA_N)
            comptime _v_rows_per_stage = (_v_num_mmas_v * Self.num_k_mmas2)

            var reg_loader = RegTileLoader[
                Self.kv_t.dtype,
                row_major[_v_thr_rows, _v_thr_cols](),
                Self.num_threads,
            ](gmem_tile)

            var load_buf = stack_allocation[
                Self.kv_t.dtype, address_space=.LOCAL
            ](row_major[_v_rows_per_stage, _v_sw]())

            var dram_strip = gmem_tile.tile[Self.BK, Self.depth](strip_idx, 0)
            reg_loader.load(
                load_buf,
                dram_strip.vectorize[1, _v_sw](),
            )

            # Flat SMEM tile [BK, depth] — row_major stride = depth.
            comptime _v_smem_layout = row_major[Self.BK, Self.depth]()
            var v_smem = TileTensor[
                Self.kv_t.dtype,
                type_of(_v_smem_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](self.smem_ptr, _v_smem_layout)

            RegTileWriterLDS[
                row_major[_v_thr_rows, _v_thr_cols](),
                None,
                Self.num_threads,
            ].copy(
                v_smem.vectorize[1, _v_sw](),
                load_buf.vectorize[1, _v_sw](),
            )

    @always_inline
    def load_from_shared(self):
        """Load from SMEM to MMA registers."""
        comptime if Self.transpose:
            comptime if Self.kv_t.dtype.is_float8():
                # FP8 K (decode): MMA_M=16, MMA_K=32 → frag_width=8, which
                # load_lds_fragment handles with a single 8-element FP8
                # load per lane (use_fp8_split only fires at MMA_K=128).
                var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
                var warp_smem = self.k_smem_block_tile[Self.WN](Int(warp_col))
                load_lds_fragment[Self.MMA_K, Self.swizzle](
                    warp_smem, self.kv_mma_op.reg_tile
                )
            else:
                # BF16 K (transpose): TiledMmaOp.load_b on a [WN, BK] SMEM
                # sub-view. Distribute+swizzle matches TensorCore.load_b
                # semantics (vector-granularity) and handles the plain
                # row-major [BN, BK] SMEM that MHA's DRAM→LDS writes.
                var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
                var warp_tile = self.k_smem_block_tile[Self.WN](Int(warp_col))

                comptime for kg in range(Self.num_k_mmas2):
                    TiledMmaOp[
                        get_accum_type[Self.kv_t.dtype](),
                        Self.kv_t.dtype,
                        Self.mma_shape,
                        transpose_b=True,
                    ].load_b[swizzle=Self.swizzle](
                        warp_tile, self.get_mma_tile[Int(kg)](), Int(kg)
                    )
        else:
            # V (non-transpose): MFMA B non-transpose register layout.
            #
            # warp_tile is a [BK, warp_depth] sub-view of [BK, depth] SMEM,
            # so its row stride is `depth` (NOT `warp_depth`) — the tile is
            # strided, not dense. TileTensor's `vectorize[simd, 1]` tracks
            # only a scalar `element_size` and loses the element-layout
            # stride, so a subsequent `distribute` emits a contiguous
            # `load[width=simd]` that reads the wrong bytes on a strided
            # tile. LayoutTensor's `vectorize` preserves the element
            # layout via `zipped_divide` and iterates element-by-element
            # in `copy_from`, which is why the LayoutTensor path works.
            #
            # The MFMA B non-transpose layout is BLOCK-distributed:
            # lane (tr, tc) with tr = lane // MMA_N, tc = lane % MMA_N
            # holds `input_frag_size` CONSECUTIVE K-rows at column tc.
            # (`distribute_with_offset[row_major[WARP_SIZE/MMA_N, MMA_N]]`
            # would give CYCLIC distribution — rows [tr, tr+4, tr+8, …] —
            # which is the wrong register permutation for MFMA.) Emit
            # the block load explicitly with scalar strided reads.
            comptime _v_warp_depth = Self.depth // Self._num_warps_n
            comptime _v_num_mmas_v = ceildiv(_v_warp_depth, Self.MMA_N)
            comptime _v_k_rows = Self.MMA_K
            comptime _v_smem_layout = row_major[Self.BK, Self.depth]()

            var warp_col = umod(Int(self.warp_id), Self._num_warps_n)
            var smem_tile = TileTensor[
                Self.kv_t.dtype,
                type_of(_v_smem_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](self.smem_ptr, _v_smem_layout)
            var warp_tile = smem_tile.tile[Self.BK, _v_warp_depth](
                0, Int(warp_col)
            )

            var reg_vec = self.kv_mma_op.reg_tile.vectorize[
                1, Self.input_frag_size
            ]()

            var tr, tc = divmod(Int(lane_id()), Self.MMA_N)

            comptime for kg in range(Self.num_k_mmas2):
                comptime for i in range(_v_num_mmas_v):
                    # mma_view: [_v_k_rows, MMA_N] at (kg, i) within
                    # warp_tile; row stride = depth (strided tile).
                    var mma_view = warp_tile.tile[_v_k_rows, Self.MMA_N](kg, i)
                    var frag = SIMD[Self.kv_t.dtype, Self.input_frag_size]()
                    comptime for r in range(Self.input_frag_size):
                        frag[r] = mma_view[tr * Self.input_frag_size + r, tc][0]
                    reg_vec[kg * _v_num_mmas_v + i, 0] = frag

    @always_inline
    def get_mma_tile[
        k_mma_idx: Int,
    ](self) -> TileTensor[
        Self.kv_t.dtype,
        type_of(row_major[Self.num_mmas, Self.input_frag_size]()),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]:
        """Get register tile for one k_mma group within the single strip.

        Parameters:
            k_mma_idx: Index of the MFMA tile along the K dimension
                within the strip.
        """
        return self.kv_mma_op.mma_tile_at[0, k_mma_idx]()


# ===----------------------------------------------------------------------=== #
# DecodeKVBuffer — register-staged buffer for the decode mirror path
# ===----------------------------------------------------------------------=== #


trait KVBufferConfig:
    """Tiling configuration shared by the K and V decode buffer roles.

    Defines the warp-tile and block-tile dimensions, the MMA iteration
    axis used to walk the DRAM tile, and the warp-tile coordinate lookup
    that `DecodeKVBuffer` uses to select its SMEM sub-view per warp.
    """

    comptime wsize: Int
    comptime wtile_dim0: Int
    comptime wtile_dim1: Int

    comptime btile_dim0: Int
    comptime btile_dim1: Int

    comptime iterator_axis: Int

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        ...


@fieldwise_init
struct KBufferConfig[BN: Int, BK: Int, WN: Int](KVBufferConfig):
    """K-role decode buffer config that iterates DRAM along the column axis.

    The warp tile is `[WN, BK]` (key-block width by strip width), the block
    tile is `[BN, BK]`, and `iterator_axis = 1` advances `tile_idx` across
    column strips of the DRAM tile.

    Parameters:
        BN: Number of keys per block tile (extent along the key axis).
        BK: Strip width along the MMA K dimension.
        WN: Number of keys per warp tile.
    """

    comptime wsize = Self.wtile_dim0
    comptime wtile_dim0 = Self.WN
    comptime wtile_dim1 = Self.BK

    comptime btile_dim0 = Self.BN
    comptime btile_dim1 = Self.BK

    comptime iterator_axis = 1

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        return IndexList[2](warp_col, 0)


@fieldwise_init
struct VBufferConfig[BN: Int, BK: Int, WN: Int, depth: Int](KVBufferConfig):
    """V-role decode buffer config that iterates DRAM along the row axis.

    The warp tile is `[BK, depth // num_warps_n]` (strip width by per-warp
    depth slice), the block tile is `[BK, depth]`, and `iterator_axis = 0`
    advances `tile_idx` across row strips of the DRAM tile.

    Parameters:
        BN: Number of keys per block tile (extent along the key axis).
        BK: Strip width along the MMA K dimension.
        WN: Number of keys per warp tile.
        depth: Per-head depth of the KV cache (head dimension).
    """

    comptime wsize = Self.wtile_dim1
    comptime wtile_dim0 = Self.BK
    comptime wtile_dim1 = Self.depth // (Self.BN // Self.WN)

    comptime btile_dim0 = Self.BK
    comptime btile_dim1 = Self.depth

    comptime iterator_axis = 0

    @staticmethod
    @always_inline
    def get_wtile_coord() -> IndexList[2]:
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        return IndexList[2](0, warp_col)


struct DecodeKVBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    //,
    config: KVBufferConfig,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
]:
    """Double-buffered register-staged KV buffer for the AMD decode mirror path.

    Stages DRAM tiles through a register `load_tile` (optionally multi-stage)
    before copying to SMEM and loading into the MFMA `mma_tile`. The
    `KVBufferConfig` parameter selects the K (transpose) or V
    (non-transpose) tiling, iterator axis, and warp-tile coordinate so a
    single implementation serves both operands.

    Parameters:
        dtype: Element type of the KV cache tiles (inferred).
        kv_tile_layout: Runtime layout of the DRAM KV tile used to build
            `gmem_tile` (inferred).
        config: K or V role config selecting warp tile, block tile,
            iterator axis, and warp-tile coordinate.
        tensor_core_mma: MFMA descriptor (shape, group size, transpose_b,
            out type) driving SMEM-to-register loads.
        swizzle: LDS bank-conflict swizzle for SMEM loads and stores, or
            `None`.
        BN: Number of keys per DRAM tile (block tile extent along the
            key axis).
        WN: Number of keys per warp tile (warp tile extent along the key
            axis).
        BK: Strip width along the MMA K dimension of one DRAM-to-SMEM
            load.
        depth: Per-head depth of the KV cache (head dimension).
        num_threads: Number of threads in the block driving DMA and
            register distribution.
        num_stages: Number of register buffer stages for double buffering
            (defaults to 1).
        token_gen: Selects the token-generation thread layout for DMA
            distribution (defaults to `False`).
    """

    comptime _dtype = Self.dtype
    comptime _num_stages = Self.num_stages
    comptime MMA_N = Self.tensor_core_mma.shape[1]
    comptime MMA_K = Self.tensor_core_mma.shape[2]
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(Self.config.wsize, Self.MMA_N)

    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime simd_width = simd_width_of[Self.dtype]()

    # Thread layout for DRAM→register and register→SMEM distribution.
    # token_gen uses a layout matching the vectorized smem tile shape;
    # non-token_gen uses the standard (num_threads//4, 4) grid.
    comptime _btile_dim0 = Self.config.btile_dim0
    comptime _btile_dim1 = Self.config.btile_dim1
    comptime _thread_rows = (
        min(
            Self.num_threads,
            (Self._btile_dim0 * Self._btile_dim1) // Self.simd_width,
        )
        * Self.simd_width
        // Self._btile_dim1
    ) if Self.token_gen else Self.num_threads // 4
    comptime _thread_cols = (
        Self._btile_dim1 // Self.simd_width
    ) if Self.token_gen else 4

    # TileTensor register storage for DMA staging.
    comptime _rows_per_stage = Self.num_mmas * Self.num_k_tiles
    comptime _load_rows = Self.num_stages * Self._rows_per_stage
    comptime _load_layout = row_major[Self._load_rows, Self.simd_width]()
    comptime LoadTile = TileTensor[
        Self.dtype,
        type_of(Self._load_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var load_tile: Self.LoadTile

    # TileTensor register storage for MMA operand.
    comptime _mma_rows = Self.num_mmas
    comptime _mma_cols = Self.simd_width
    comptime _mma_layout = row_major[Self._mma_rows, Self._mma_cols]()
    comptime MmaTile = TileTensor[
        Self.dtype,
        type_of(Self._mma_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var mma_tile: Self.MmaTile

    comptime wtile_dim0 = Self.config.wtile_dim0
    comptime wtile_dim1 = Self.config.wtile_dim1

    # TiledMmaOp for SMEM→register loads.
    comptime _TiledMma = TiledMmaOp[
        out_type=Self.tensor_core_mma.out_type,
        in_type=Self.dtype,
        shape=Self.tensor_core_mma.shape,
        transpose_b=Self.tensor_core_mma.transpose_b,
    ]

    # SMEM TileTensor (stored once, used by copy_to_shared and load_from_shared).
    comptime _smem_layout = row_major[Self._btile_dim0, Self._btile_dim1]()
    comptime SmemTile = TileTensor[
        Self.dtype,
        type_of(Self._smem_layout),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var smem_tile: Self.SmemTile

    # DRAM tile and loader.
    comptime GmemTileType = TileTensor[
        Self.dtype, Self.kv_tile_layout, ImmutAnyOrigin
    ]
    # num_threads overridden: block has more threads than the load layout.
    comptime RegLoaderType = RegTileLoader[
        Self.dtype,
        row_major[Self._thread_rows, Self._thread_cols](),
        Self.num_threads,
    ]

    @__allow_legacy_any_origin_fields
    var gmem_tile: Self.GmemTileType
    var reg_loader: Self.RegLoaderType
    var tile_idx: Int
    var load_tile_id: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: Self.GmemTileType,
        smem_tile: Self.SmemTile,
    ):
        self.load_tile = stack_allocation[Self.dtype, address_space=.LOCAL](
            Self._load_layout
        )
        self.mma_tile = stack_allocation[Self.dtype, address_space=.LOCAL](
            Self._mma_layout
        )
        self.smem_tile = smem_tile
        self.gmem_tile = gmem_tile
        self.reg_loader = Self.RegLoaderType(gmem_tile)
        self.tile_idx = 0
        self.load_tile_id = 0

    @always_inline
    def load_from_dram(
        mut self,
    ):
        # Build per-iteration DRAM sub-tile on-the-fly.
        var row_idx = 0 if Self.config.iterator_axis == 1 else self.tile_idx
        var col_idx = self.tile_idx if Self.config.iterator_axis == 1 else 0
        var src = self.gmem_tile.tile[Self._btile_dim0, Self._btile_dim1](
            row_idx, col_idx
        )
        var dst = self.load_tile.tile[Self._rows_per_stage, Self.simd_width](
            self.load_tile_id, 0
        )
        self.reg_loader.load(
            dst,
            src.vectorize[1, Self.simd_width](),
        )
        self.tile_idx += 1
        self.load_tile_id = (self.load_tile_id + 1) % Self.num_stages

    @always_inline
    def get_mma_tile(
        self,
    ) -> TileTensor[
        Self.dtype,
        type_of(row_major[Self._mma_rows, Self._mma_cols]()),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]:
        return rebind[
            TileTensor[
                Self.dtype,
                type_of(row_major[Self._mma_rows, Self._mma_cols]()),
                MutAnyOrigin,
                address_space=.LOCAL,
            ]
        ](self.mma_tile)

    @always_inline
    def copy_to_shared[
        tile_id: Int = 0
    ](self,):
        RegTileWriterLDS[
            row_major[Self._thread_rows, Self._thread_cols](),
            Self.swizzle,
            Self.num_threads,
        ].copy(
            self.smem_tile.vectorize[1, Self.simd_width](),
            self.load_tile.tile[Self._rows_per_stage, Self.simd_width](
                tile_id, 0
            ).vectorize[1, Self.simd_width](),
        )

    @always_inline
    def load_from_shared[
        k_mma: Int,
    ](self):
        var wtile_coord0 = Self.config.get_wtile_coord()[0]
        var wtile_coord1 = Self.config.get_wtile_coord()[1]
        var warp_tile = self.smem_tile.tile[Self.wtile_dim0, Self.wtile_dim1](
            wtile_coord0, wtile_coord1
        )

        comptime if Self.tensor_core_mma.transpose_b:
            # K (transpose_b=True): warp_tile is a contiguous [WN, BK]
            # sub-view, TiledMmaOp.load_b handles swizzle + distribute.
            Self._TiledMma.load_b[swizzle=Self.swizzle](
                warp_tile, self.mma_tile, k_mma
            )
        else:
            # V (transpose_b=False): warp_tile is a strided [BK, warp_depth]
            # sub-view of [BK, depth] (row stride = depth). TileTensor
            # vectorize drops the element-layout stride, so emit the MFMA
            # B non-transpose BLOCK distribution explicitly: lane
            # (tr, tc) with tr = lane // MMA_N, tc = lane % MMA_N holds
            # `simd_width` consecutive K-rows at column tc.
            comptime k_rows = (Self.MMA_K * Self.tensor_core_mma.group_size)
            var tr, tc = divmod(Int(lane_id()), Self.MMA_N)
            var reg_vec = self.mma_tile.vectorize[1, Self.simd_width]()

            comptime for i in range(Self.num_mmas):
                var mma_view = warp_tile.tile[k_rows, Self.MMA_N](k_mma, i)
                var frag = SIMD[Self.dtype, Self.simd_width]()
                comptime for r in range(Self.simd_width):
                    frag[r] = mma_view[tr * Self.simd_width + r, tc][0]
                reg_vec[i, 0] = frag


comptime KBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
] = DecodeKVBuffer[
    dtype=dtype,
    kv_tile_layout=kv_tile_layout,
    config=KBufferConfig[BN, BK, WN],
    tensor_core_mma=tensor_core_mma,
    swizzle=swizzle,
    BN=BN,
    WN=WN,
    BK=BK,
    depth=depth,
    num_threads=num_threads,
    num_stages=num_stages,
    token_gen=token_gen,
]

comptime VBuffer[
    dtype: DType,
    kv_tile_layout: TensorLayout,
    tensor_core_mma: TiledTensorCore,
    swizzle: Optional[Swizzle],
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    token_gen: Bool = False,
] = DecodeKVBuffer[
    dtype=dtype,
    kv_tile_layout=kv_tile_layout,
    config=VBufferConfig[BN, BK, WN, depth],
    tensor_core_mma=tensor_core_mma,
    swizzle=swizzle,
    BN=BN,
    WN=WN,
    BK=BK,
    depth=depth,
    num_threads=num_threads,
    num_stages=num_stages,
    token_gen=token_gen,
]
