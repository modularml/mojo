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
"""K, V, Q, P, and Output buffers for RDNA Wave32 attention kernels.

Wave32 WMMA fragment geometry:
  A/B per lane = 16 elements (full K; lanes 0-15 unique, 16-31 replicate)
  C/D per lane = 8 elements (M*N / WARP = 256 / 32)
  Lane->C/D map: lane l, elem v -> D[row = v*2 + l//16, col = l%16]
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.math.uutils import umod
from std.sys import simd_width_of

from max.gpu import lane_id, warp_id as get_warp_id
from layout import TensorLayout, TileTensor
from layout.tensor_core import TiledTensorCore
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from structured_kernels.amd_tile_io import RegTileLoader

from std.utils import IndexList

from .config import RDNA_MMA_M, RDNA_MMA_N, RDNA_MMA_K
from .utils import get_warp_coords, pad

comptime RDNA_WARP_SIZE = 32
comptime RDNA_AB_FRAG_SIZE = 16
comptime RDNA_CD_FRAG_SIZE = 8


# ===----------------------------------------------------------------------=== #
# K buffer — DRAM (BN x depth) -> LDS (BN x BK strips) -> register fragments
# ===----------------------------------------------------------------------=== #


struct KBufferRDNA[
    cache_dtype: DType,
    gmem_layout: TensorLayout,
    //,
    tensor_core_mma: TiledTensorCore,
    BN: Int,
    WN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
]:
    """K buffer: holds a (BN, depth) DRAM tile reference, a register
    `load_tile` that staggers DMA across BK strips, an `mma_tile` for the
    current K fragment, and a (BN, BK) LDS region for the staged strip.

    Parameters:
        cache_dtype: Element type of the K cache tiles in DRAM and LDS
            (inferred).
        gmem_layout: `TensorLayout` of the DRAM K tile (inferred).
        tensor_core_mma: `TiledTensorCore` describing the MMA instruction
            used for the QK product.
        BN: Total N (key) tile width across all warps, in elements.
        WN: N tile width owned by each warp, in elements.
        BK: K strip width loaded per pipeline stage, in elements.
        depth: Total head depth of the K matrix in DRAM, in elements.
        num_threads: Number of threads participating in DRAM-to-register
            loads.
        num_stages: Number of software-pipelining register stages (defaults
            to 1).
    """

    comptime _dtype = Self.cache_dtype
    comptime _num_stages = Self.num_stages
    comptime MMA_M = RDNA_MMA_M
    comptime MMA_N = RDNA_MMA_N
    comptime MMA_K = RDNA_MMA_K
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_mmas = ceildiv(Self.WN, Self.MMA_N)

    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime simd_width = simd_width_of[Self.cache_dtype]()

    # Block-scope thread distribution for DRAM->reg: each thread loads
    # `simd_width` contiguous columns; 4 cols per lane-row gives
    # num_threads/4 row-positions.
    comptime _thread_rows = Self.num_threads // 4
    comptime _thread_cols = 4

    comptime _per_stage_rows = Self.num_mmas * Self.num_k_tiles
    comptime _load_rows = Self.num_stages * Self._per_stage_rows
    comptime load_layout = row_major[Self._load_rows, Self.simd_width]()
    comptime LoadTileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.load_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var load_tile: Self.LoadTileType

    comptime mma_tile_layout = row_major[Self.num_mmas, RDNA_AB_FRAG_SIZE]()
    comptime MMATileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.mma_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var mma_tile: Self.MMATileType

    # K SMEM holds one BN x BK strip at a time; double-buffering happens
    # in registers (num_stages-deep `load_tile`), not in SMEM.
    comptime smem_layout = row_major[Self.BN, Self.BK]()
    comptime SmemTileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.smem_layout),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var smem_tile: Self.SmemTileType

    # DRAM tile + strip iterator state.
    @__allow_legacy_any_origin_fields
    var gmem_tile: TileTensor[
        Self.cache_dtype, Self.gmem_layout, ImmutAnyOrigin
    ]
    var strip_idx: Int
    var load_tile_id: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[
            Self.cache_dtype, Self.gmem_layout, ImmutAnyOrigin
        ],
        shared_ptr: UnsafePointer[
            Scalar[Self.cache_dtype], MutAnyOrigin, address_space=.SHARED
        ],
    ):
        self.load_tile = tt_stack_allocation[
            Self.cache_dtype, address_space=.LOCAL
        ](Self.load_layout)
        self.mma_tile = tt_stack_allocation[
            Self.cache_dtype, address_space=.LOCAL
        ](Self.mma_tile_layout)
        self.smem_tile = Self.SmemTileType(shared_ptr, Self.smem_layout)
        self.gmem_tile = gmem_tile
        self.strip_idx = 0
        self.load_tile_id = 0

    @always_inline
    @staticmethod
    def get_dtype() -> DType:
        return Self._dtype

    @always_inline
    def load_from_dram(mut self):
        """Load the next BK strip of K from DRAM into the staging
        register slot. The gmem_tile's runtime row count drives SRD
        OOB clamping for unaligned tail tiles."""
        var strip = self.gmem_tile.tile[Self.BN, Self.BK](0, self.strip_idx)
        var dst = self.load_tile.tile[Self._per_stage_rows, Self.simd_width](
            self.load_tile_id, 0
        )
        var loader = RegTileLoader[
            Self.cache_dtype,
            row_major[Self._thread_rows, Self._thread_cols](),
            num_threads=Self.num_threads,
            warp_scope=False,
        ](strip, bounds_from=self.gmem_tile)
        loader.load(dst, strip.vectorize[1, Self.simd_width]())
        self.strip_idx += 1
        self.load_tile_id = (self.load_tile_id + 1) % Self.num_stages

    @always_inline
    def get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    def copy_to_shared[tile_id: Int = 0](self):
        """Write the staging register slot `tile_id` to LDS, distributing
        the (BN, BK) tile across threads using the same row_major(
        _thread_rows, _thread_cols) layout as `load_from_dram`.

        Parameters:
            tile_id: Index of the pipeline-stage register slot to flush to
                LDS (defaults to 0).
        """
        var lane = lane_id()
        var warp = get_warp_id()
        var tid = warp * RDNA_WARP_SIZE + lane

        comptime num_busy = Self._thread_rows * Self._thread_cols
        if Int(tid) >= num_busy:
            return

        var tr = Int(tid) // Self._thread_cols
        var tc = Int(tid) % Self._thread_cols

        var src = self.load_tile.tile[Self._per_stage_rows, Self.simd_width](
            tile_id, 0
        )

        comptime for i in range(Self._per_stage_rows):
            var smem_row = tr + i * Self._thread_rows
            var smem_col = tc * Self.simd_width
            comptime for j in range(Self.simd_width):
                self.smem_tile[smem_row, smem_col + j] = src[i, j]

    @always_inline
    def load_from_shared[k_mma: Int](self):
        """SMEM->fragment, wave-cooperative.

        K is the A operand under swap_a_b. RDNA WMMA A maps
        a_frag[v] = A[lane % 16, v], so lane selects key (K row) and
        element selects depth (K column).

        Parameters:
            k_mma: Index of the MMA tile along the K axis within the
                current BK strip.
        """
        var warp_col = get_warp_coords[Self.BN, Self.WN]()[1]
        var warp_tile = self.smem_tile.tile[Self.WN, Self.BK](warp_col, 0)
        var lane = umod(lane_id(), 16)

        comptime for m in range(Self.num_mmas):
            comptime for k in range(RDNA_AB_FRAG_SIZE):
                self.mma_tile[m, k] = warp_tile[
                    m * Self.MMA_N + Int(lane), k_mma * Self.MMA_K + k
                ][0]


# ===----------------------------------------------------------------------=== #
# V buffer — DRAM (BK x depth) -> LDS (depth x BK transposed) -> fragments
# ===----------------------------------------------------------------------=== #


struct VBufferRDNA[
    cache_dtype: DType,
    gmem_layout: TensorLayout,
    //,
    tensor_core_mma: TiledTensorCore,
    BN: Int,
    BK: Int,
    depth: Int,
    num_threads: Int,
    num_stages: Int = 1,
    num_warps_n: Int = 1,
]:
    """V buffer with transpose-on-LDS-write.

    V is read from DRAM in row-major (BK x depth) chunks but written to
    LDS as `pad(depth) x BK` blocked in `simd_width`-wide column groups,
    so the per-warp depth slice reads contiguously during the PV MMA.

    Parameters:
        cache_dtype: Element type of the V cache tiles in DRAM and LDS
            (inferred).
        gmem_layout: `TensorLayout` of the DRAM V tile (inferred).
        tensor_core_mma: `TiledTensorCore` describing the MMA instruction
            used for the PV product.
        BN: Total N (key) tile width across all warps, in elements.
        BK: K strip width loaded per pipeline stage, in elements.
        depth: Total head depth of the V matrix in DRAM, in elements.
        num_threads: Number of threads participating in DRAM-to-register
            loads.
        num_stages: Number of software-pipelining register stages (defaults
            to 1).
        num_warps_n: Number of warps along the N (key) axis (defaults to 1).
    """

    comptime _dtype = Self.cache_dtype
    comptime _num_stages = Self.num_stages
    comptime simd_width = simd_width_of[Self.cache_dtype]()
    comptime num_repeats = Self.BK // Self.simd_width

    comptime MMA_M = RDNA_MMA_M
    comptime MMA_K = RDNA_MMA_K
    comptime num_k_tiles = ceildiv(
        Self.BK, Self.MMA_K * Self.tensor_core_mma.group_size
    )
    comptime num_depth_tiles = Self.depth // Self.MMA_M
    comptime warp_depth_tiles = Self.depth // Self.num_warps_n // Self.MMA_M

    comptime depth_tile_size = min(Self.depth, 128)
    comptime load_width = 4 if Self.depth == 64 else Self.simd_width

    comptime loads_per_thread_per_depth_tile = (
        Self.depth_tile_size * Self.BK
    ) // (Self.load_width * Self.num_threads)

    # Register staging tile.
    comptime _load_rows = (
        Self.loads_per_thread_per_depth_tile
        * (Self.depth // Self.depth_tile_size)
    ) * Self.num_stages
    comptime load_layout = row_major[Self._load_rows, Self.load_width]()
    comptime LoadTileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.load_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var load_tile: Self.LoadTileType

    comptime mma_tile_layout = row_major[
        Self.warp_depth_tiles, RDNA_AB_FRAG_SIZE
    ]()
    comptime MMATileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.mma_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var mma_tile: Self.MMATileType

    # V SMEM: num_repeats column-blocks of (pad(depth) × simd_width)
    # stacked along the row axis. Encoded as a flat row_major layout
    # because the transpose-write needs explicit per-block offsets.
    comptime _smem_padded_depth = pad[
        Self.cache_dtype, Self.depth, Self.depth
    ]()
    comptime smem_layout = row_major[
        Self._smem_padded_depth * Self.num_repeats, Self.simd_width
    ]()
    comptime SmemTileType = TileTensor[
        Self.cache_dtype,
        type_of(Self.smem_layout),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var smem_tile: Self.SmemTileType

    # The DRAM tile spans (BK x depth) per call; the buffer iterates
    # `strip_idx` over BK strips internally on each load_from_dram.
    @__allow_legacy_any_origin_fields
    var gmem_tile: TileTensor[
        Self.cache_dtype, Self.gmem_layout, ImmutAnyOrigin
    ]
    var strip_idx: Int
    var current_stage: Int
    var remaining_rows: Int

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[
            Self.cache_dtype, Self.gmem_layout, ImmutAnyOrigin
        ],
        shared_ptr: UnsafePointer[
            Scalar[Self.cache_dtype], MutAnyOrigin, address_space=.SHARED
        ],
        total_rows: OptionalReg[Int] = None,
    ):
        comptime assert Self.depth in (
            64,
            128,
            256,
            512,
        ), "depth must be 64, 128, 256, or 512"

        self.load_tile = tt_stack_allocation[
            Self.cache_dtype, address_space=.LOCAL
        ](Self.load_layout)
        self.mma_tile = tt_stack_allocation[
            Self.cache_dtype, address_space=.LOCAL
        ](Self.mma_tile_layout)
        self.smem_tile = Self.SmemTileType(shared_ptr, Self.smem_layout)
        self.gmem_tile = gmem_tile
        self.strip_idx = 0
        self.current_stage = 0
        self.remaining_rows = total_rows.value() if total_rows else Int.MAX

    @always_inline
    @staticmethod
    def get_dtype() -> DType:
        return Self._dtype

    @always_inline
    @staticmethod
    def pad[dim: Int]() -> Int:
        return pad[Self.cache_dtype, Self.depth, dim]()

    @always_inline
    def load_from_dram(mut self):
        """Load the next BK strip of V from DRAM into the staging
        register slot. Per-thread chunks are bounds-clamped via
        `remaining_rows` for the unaligned tail tile."""
        var strip = self.gmem_tile.tile[Self.BK, Self.depth](self.strip_idx, 0)
        var stride = Self.gmem_layout.static_stride[0]
        var warp = get_warp_id()

        var load_tile = self.load_tile.tile[
            Self.loads_per_thread_per_depth_tile
            * (Self.depth // Self.depth_tile_size),
            Self.load_width,
        ](self.current_stage, 0)

        comptime num_warps = Self.num_threads // RDNA_WARP_SIZE
        comptime rows_per_warp = Self.BK // num_warps
        comptime threads_per_row = RDNA_WARP_SIZE // rows_per_warp
        comptime depth_per_thread = Self.depth_tile_size // threads_per_row

        var lane = lane_id()
        var thread_row, thread_col = divmod(lane, threads_per_row)

        var src_row = warp * rows_per_warp + thread_row
        var tile_valid_rows = min(Self.BK, self.remaining_rows)

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            comptime for i in range(Self.loads_per_thread_per_depth_tile):
                var dst_idx = (
                    i + depth_idx * Self.loads_per_thread_per_depth_tile
                )
                if Int(src_row) < tile_valid_rows:
                    var src_col = (
                        thread_col * depth_per_thread + i * Self.load_width
                    )
                    src_col += depth_idx * Self.depth_tile_size
                    var offset = Int(src_row) * stride + Int(src_col)
                    var data = (strip.ptr + offset).load[
                        width=Self.load_width
                    ]()
                    comptime for j in range(Self.load_width):
                        load_tile[dst_idx, j] = data[j]
                else:
                    comptime for j in range(Self.load_width):
                        load_tile[dst_idx, j] = 0

        self.strip_idx += 1
        self.remaining_rows -= Self.BK
        self.current_stage = (self.current_stage + 1) % Self.num_stages

    @always_inline
    def get_mma_tile(self) -> Self.MMATileType:
        return self.mma_tile

    @always_inline
    def copy_to_shared[tile_id: Int = 0](self):
        """V transpose-on-write: smem[depth_pos, seq_pos] from
        load_tile[seq_pos, depth_pos] (with depth_pos blocked into
        `simd_width`-wide column groups along the SMEM row axis).

        Parameters:
            tile_id: Index of the pipeline-stage register slot to flush to
                LDS (defaults to 0).
        """
        var warp = get_warp_id()
        var lane = lane_id()

        comptime num_warps = Self.num_threads // RDNA_WARP_SIZE
        comptime rows_per_warp = Self.BK // num_warps
        comptime threads_per_row = RDNA_WARP_SIZE // rows_per_warp
        comptime depth_per_thread = Self.depth_tile_size // threads_per_row

        var thread_row, thread_col = divmod(lane, threads_per_row)

        var load_tile = self.load_tile.tile[
            Self.loads_per_thread_per_depth_tile
            * (Self.depth // Self.depth_tile_size),
            Self.load_width,
        ](tile_id, 0)

        # smem_col_abs is the seq position (the V row in DRAM).
        # smem_chunk indexes the simd_width-wide col block; smem_col is
        # within-block.
        var smem_col_abs = warp * rows_per_warp + thread_row
        var smem_chunk, smem_col = divmod(smem_col_abs, Self.simd_width)

        comptime for depth_idx in range(Self.depth // Self.depth_tile_size):
            # Physical row in the flat `[padded_depth * num_repeats,
            # simd_width]` SMEM: smem_chunk picks the simd_width-wide
            # col block, depth_idx picks the depth-tile within it.
            var depth_block_base = (
                smem_chunk * Self._smem_padded_depth
                + depth_idx * Self.pad[Self.depth_tile_size]()
            )

            comptime for i in range(Self.loads_per_thread_per_depth_tile):
                var smem_row_base = (
                    thread_col * depth_per_thread + i * Self.load_width
                )
                comptime for j in range(Self.load_width):
                    var smem_row = Int(depth_block_base) + (
                        Int(smem_row_base) + j
                    )
                    var val = load_tile[
                        i + depth_idx * Self.loads_per_thread_per_depth_tile, j
                    ]
                    self.smem_tile[smem_row, Int(smem_col)] = val

    @always_inline
    def load_from_shared[k_mma: Int](self):
        """SMEM->fragment, wave-cooperative.

        V is transposed in SMEM as `[depth, key]`. Under swap_a_b V is
        the A operand, so lane selects the depth row and element
        selects the key column.

        Parameters:
            k_mma: Index of the MMA tile along the K (key) axis within the
                current BK strip.
        """
        var lane = umod(lane_id(), 16)
        var warp_n_idx = get_warp_id() % Self.num_warps_n
        var depth_offset = warp_n_idx * Self.warp_depth_tiles * Self.MMA_M

        comptime for depth_idx in range(Self.warp_depth_tiles):
            var global_depth_pos = (
                Int(depth_offset) + depth_idx * Self.MMA_M + Int(lane)
            )
            var dtile_idx, pos_in_dtile = divmod(
                global_depth_pos, Self.depth_tile_size
            )
            var depth_row = (
                dtile_idx * Self.pad[Self.depth_tile_size]() + pos_in_dtile
            )

            comptime for j in range(RDNA_AB_FRAG_SIZE):
                # Block 0 holds keys [k_mma*2*simd, k_mma*2*simd+simd);
                # block 1 holds keys [k_mma*2*simd+simd, k_mma*2*simd+2*simd).
                comptime if j < Self.simd_width:
                    var block_row = (
                        depth_row + (k_mma * 2) * Self._smem_padded_depth
                    )
                    self.mma_tile[depth_idx, j] = self.smem_tile[block_row, j][
                        0
                    ]
                else:
                    var block_row = (
                        depth_row + (k_mma * 2 + 1) * Self._smem_padded_depth
                    )
                    self.mma_tile[depth_idx, j] = self.smem_tile[
                        block_row, j - Self.simd_width
                    ][0]


# ===----------------------------------------------------------------------=== #
# Q register buffer — DRAM (WM x depth) -> registers (per-strip BK frags)
# ===----------------------------------------------------------------------=== #


struct QRegisterBufferRDNA[
    dtype: DType,
    mma_shape: IndexList[3],
    k_group_size: Int,
    WM: Int,
    WN: Int,
    BN: Int,
    BK: Int,
    depth: Int,
]:
    """Q register buffer: loads each warp's (WM, depth) Q sub-tile into
    BK-strip MMA fragments at construction.

    Parameters:
        dtype: Element type of the Q tensor and register fragments.
        mma_shape: `(M, N, K)` shape of a single MMA instruction as an
            `IndexList[3]`.
        k_group_size: Grouping factor for the MMA K dimension.
        WM: M (query) tile height owned by each warp, in elements.
        WN: N tile width owned by each warp, in elements.
        BN: Total N (key) tile width across all warps, in elements.
        BK: K strip width loaded per pipeline stage, in elements.
        depth: Total head depth of the Q matrix in DRAM, in elements.
    """

    comptime reg_dtype = Self.dtype
    comptime mma_dtype = Self.dtype
    comptime simd_width = simd_width_of[Self.dtype]()
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_mmas = ceildiv(Self.WM, Self.MMA_M)
    comptime num_k_tiles = ceildiv(Self.BK, Self.MMA_K * Self.k_group_size)

    comptime rdna_frag_size = RDNA_AB_FRAG_SIZE
    comptime num_tiles = Self.depth // Self.BK

    comptime reg_tile_layout = row_major[
        Self.num_mmas * Self.num_k_tiles * Self.num_tiles, Self.rdna_frag_size
    ]()
    comptime RegisterTileType = TileTensor[
        Self.dtype,
        type_of(Self.reg_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    comptime mma_tile_layout = row_major[Self.num_mmas, Self.rdna_frag_size]()
    comptime MMATileType = TileTensor[
        Self.dtype,
        type_of(Self.mma_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var reg_tile: Self.RegisterTileType

    @staticmethod
    @always_inline
    def get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    def __init__[
        q_layout: TensorLayout
    ](
        out self,
        tensor: TileTensor[Self.dtype, q_layout, ImmutAnyOrigin],
        valid_rows: Int,
    ):
        """Load each warp's Q sub-tile from DRAM into register MMA
        fragments.

        Parameters:
            q_layout: `TensorLayout` of the Q tensor in DRAM (inferred).

        Args:
            tensor: DRAM Q tile in `q_layout`.
            valid_rows: Q row bound for OOB clamping, equal to the group
                size for decode or the clamped sequence tile size for
                prefill.
        """
        self.reg_tile = tt_stack_allocation[Self.dtype, address_space=.LOCAL](
            Self.reg_tile_layout
        )

        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_base_row = warp_row * Self.WM

        var lane = lane_id()
        # RDNA WMMA B register: b_frag[v] = B[v, lane%16].
        # With swap_a_b, Q goes to hardware B. We need B = Q^T[depth, seq].
        var row_in_first_mma = umod(lane, 16)

        comptime stride = q_layout.static_stride[0]
        var warp_offset = warp_base_row * stride

        comptime for tile_idx in range(Self.num_tiles):
            var depth_offset = tile_idx * Self.BK

            comptime for k_mma in range(Self.num_k_tiles):
                var k_offset = depth_offset + k_mma * Self.MMA_K

                comptime for m_mma in range(Self.num_mmas):
                    var row = Int(row_in_first_mma) + m_mma * 16
                    var row_offset = row * stride

                    var frag_idx = (
                        tile_idx * Self.num_k_tiles * Self.num_mmas
                        + k_mma * Self.num_mmas
                        + m_mma
                    )

                    if row < (valid_rows - Int(warp_base_row)):
                        var offset0 = warp_offset + row_offset + k_offset
                        var data0 = (tensor.ptr + offset0).load[
                            width=Self.simd_width
                        ]()
                        var offset1 = offset0 + Self.simd_width
                        var data1 = (tensor.ptr + offset1).load[
                            width=Self.simd_width
                        ]()
                        comptime for j in range(Self.simd_width):
                            self.reg_tile[frag_idx, j] = data0[j]
                            self.reg_tile[
                                frag_idx, Self.simd_width + j
                            ] = data1[j]
                    else:
                        comptime for j in range(Self.simd_width):
                            self.reg_tile[frag_idx, j] = 0
                            self.reg_tile[frag_idx, Self.simd_width + j] = 0

    @always_inline
    def get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        """MMA fragment for the `tile_idx`-th depth tile, `k_idx`-th K
        strip within it.

        Parameters:
            tile_idx: Index of the depth tile along the depth axis.
            k_idx: Index of the K strip within the depth tile.
        """
        return rebind[Self.MMATileType](
            self.reg_tile.tile[
                Self.num_mmas * Self.num_k_tiles, Self.rdna_frag_size
            ](tile_idx, 0).tile[Self.num_mmas, Self.rdna_frag_size](k_idx, 0)
        )

    @always_inline
    def get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    def zero(self):
        _ = self.reg_tile.fill(0)


# ===----------------------------------------------------------------------=== #
# Output register buffer — accumulator for PV
# ===----------------------------------------------------------------------=== #


struct OutputRegisterBufferRDNA[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
]:
    """Output accumulator register buffer. Layout is
    (num_n_mmas * num_m_mmas, RDNA_CD_FRAG_SIZE) row_major: one row
    per MMA tile, one column per per-lane C/D register.

    Parameters:
        dtype: Element type of the output accumulator registers.
        num_m_mmas: Number of MMA tiles along the M (query) axis.
        num_n_mmas: Number of MMA tiles along the N (key) axis.
    """

    comptime reg_dtype = Self.dtype
    comptime output_frag_size = RDNA_CD_FRAG_SIZE

    comptime reg_tile_layout = row_major[
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    ]()
    comptime RegisterTileType = TileTensor[
        Self.dtype,
        type_of(Self.reg_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var reg_tile: Self.RegisterTileType

    @always_inline
    def __init__(out self):
        self.reg_tile = tt_stack_allocation[Self.dtype, address_space=.LOCAL](
            Self.reg_tile_layout
        )

    @staticmethod
    @always_inline
    def get_dtype() -> DType:
        return Self.reg_dtype

    @always_inline
    def zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    def get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile


# ===----------------------------------------------------------------------=== #
# P register buffer — softmax accumulator + SMEM staging for PV's A operand
# ===----------------------------------------------------------------------=== #


struct PRegisterBufferRDNA[
    accum_type_: DType,
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    mma_shape: IndexList[3],
    k_group_size: Int,
]:
    """P register buffer (post-softmax scores). Holds the accumulator
    in registers; `copy_to_shared` casts to dtype and writes to a
    `[BK, BM]` SMEM region that the PV phase reads back as A.

    Parameters:
        accum_type_: Element type of the accumulator registers.
        dtype: Element type used for the MMA tile and SMEM-staged P
            scores.
        BM: Total M (query/sequence) tile height, in elements.
        BN: Total N (key) tile width across all warps, in elements.
        BK: K strip width per pipeline stage, in elements.
        WM: M tile height owned by each warp, in elements.
        WN: N tile width owned by each warp, in elements.
        num_m_mmas: Number of MMA tiles along the M axis per warp.
        num_n_mmas: Number of MMA tiles along the N axis per warp.
        mma_shape: `(M, N, K)` shape of a single MMA instruction as an
            `IndexList[3]`.
        k_group_size: Grouping factor for the MMA K dimension.
    """

    comptime reg_dtype = Self.accum_type_
    comptime mma_dtype = Self.dtype
    comptime output_frag_size = RDNA_CD_FRAG_SIZE
    comptime mma_frag_size = RDNA_AB_FRAG_SIZE

    comptime mma_tile_layout = row_major[Self.num_m_mmas, Self.mma_frag_size]()
    comptime reg_tile_layout = row_major[
        Self.num_n_mmas * Self.num_m_mmas, Self.output_frag_size
    ]()

    comptime RegisterTileType = TileTensor[
        Self.accum_type_,
        type_of(Self.reg_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    comptime MMATileType = TileTensor[
        Self.mma_dtype,
        type_of(Self.mma_tile_layout),
        MutAnyOrigin,
        address_space=.LOCAL,
    ]

    @__allow_legacy_any_origin_fields
    var reg_tile: Self.RegisterTileType

    @__allow_legacy_any_origin_fields
    var shared_memory_ptr: UnsafePointer[
        Scalar[Self.dtype], MutAnyOrigin, address_space=.SHARED
    ]

    @always_inline
    def __init__(
        out self,
        shared_ptr: UnsafePointer[
            Scalar[Self.dtype], MutAnyOrigin, address_space=.SHARED
        ],
    ):
        self.reg_tile = tt_stack_allocation[
            Self.accum_type_, address_space=.LOCAL
        ](Self.reg_tile_layout)
        self.shared_memory_ptr = shared_ptr

    @always_inline
    def get_mma_tile[tile_idx: Int, k_idx: Int](self) -> Self.MMATileType:
        """Load one MMA fragment from the SMEM-staged P scores. SMEM is
        keyed as `key * BM + seq`; each lane reads its (seq, key) slot.

        Parameters:
            tile_idx: Index of the BK chunk of P scores along the N (key)
                axis.
            k_idx: Index of the MMA tile along the K (key) axis within the
                BK chunk.
        """
        var mma_reg_tile = tt_stack_allocation[
            Self.mma_dtype, address_space=.LOCAL
        ](Self.mma_tile_layout)
        var warp_row = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_base_seq = warp_row * Self.WM
        var k_key_base = k_idx * Self.mma_shape[2]
        var lane = umod(lane_id(), 16)

        comptime for m_mma in range(Self.num_m_mmas):
            var seq = warp_base_seq + m_mma * Self.mma_shape[0] + Int(lane)
            comptime for k in range(Self.mma_shape[2]):
                var key = k_key_base + k
                var smem_offset = key * Self.BM + seq
                mma_reg_tile[m_mma, k] = self.shared_memory_ptr[smem_offset]

        return mma_reg_tile

    @staticmethod
    @always_inline
    def get_dtype() -> DType:
        return Self.mma_dtype

    @always_inline
    def zero(self):
        _ = self.reg_tile.fill(0)

    @always_inline
    def get_reg_tile[stage: Int = 0](self) -> Self.RegisterTileType:
        return self.reg_tile

    @always_inline
    def copy_to_shared[chunk_idx: Int](self):
        """Cast accumulator → dtype and write the `chunk_idx`-th BK
        chunk of P to SMEM. Only the warp that owns that chunk
        participates.

        Parameters:
            chunk_idx: Index of the BK chunk of P scores to write to SMEM.
                Determines the owning warp and the local N offset within
                that warp's WN tile.
        """
        comptime reg_per_thread = Self.output_frag_size
        comptime num_warps_n = Self.BN // Self.WN
        comptime chunks_per_warp = Self.WN // Self.BK
        comptime owning_warp = chunk_idx // chunks_per_warp

        var warp_coords = get_warp_coords[Self.BN, Self.WN]()
        var warp_row = warp_coords[0]
        var warp_n_idx = warp_coords[1]

        if Int(warp_n_idx) != owning_warp:
            return

        comptime local_chunk = chunk_idx % chunks_per_warp
        comptime num_n_mmas_per_bk = Self.BK // Self.mma_shape[1]
        comptime chunk_n_start = local_chunk * num_n_mmas_per_bk

        var warp_base_seq = warp_row * Self.WM

        var lane = lane_id()
        var lane_key_group, lane_seq_offset = divmod(lane, 16)

        var reg_ptr = self.reg_tile.ptr

        comptime for m_mma in range(Self.num_m_mmas):
            comptime for n_mma_local in range(num_n_mmas_per_bk):
                comptime n_mma = chunk_n_start + n_mma_local

                var mma_seq_base = warp_base_seq + m_mma * Self.mma_shape[0]
                var mma_key_base = n_mma_local * Self.mma_shape[1]

                comptime p_reg_tile_idx = n_mma * Self.num_m_mmas + m_mma
                var reg_base_offset = p_reg_tile_idx * Self.output_frag_size

                var global_seq = mma_seq_base + Int(lane_seq_offset)

                comptime for elem in range(reg_per_thread):
                    # RDNA WMMA C/D mapping: lane l, elem v ->
                    # D[row=v*2+l//16, col=l%16].
                    # P^T[key, seq]: key = elem*2 + lane_key_group.
                    var key_in_mma = elem * 2 + Int(lane_key_group)
                    var global_key = mma_key_base + key_in_mma

                    var smem_offset = global_key * Self.BM + global_seq
                    var fp32_val = reg_ptr[reg_base_offset + elem]
                    self.shared_memory_ptr[smem_offset] = fp32_val.cast[
                        Self.dtype
                    ]()
