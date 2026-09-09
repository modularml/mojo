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
"""MMA operators for AMD matmul kernels.

Structs:
    TiledMma: Stateless MMA computation on TileTensors (mirrors
        TiledTensorCore.mma). Pure computation, no register ownership.
    MmaOp: Register ownership + SMEM loading + schedule API. Wraps
        TiledMma for per-k-tile load_frag/mma dispatch.
    QuadrantMmaOp: Owns A/B/C register tiles in LOCAL, provides quadrant
        load/compute methods for ping-pong double-buffering schedule.

Data-movement primitives (TileLoaderLDS, _load_from_lds, load_lds_fragment)
live in structured_kernels.amd_tile_io.
"""

from max.gpu.compute.mma import mma as gpu_mma
from max.gpu import lane_id, WARP_SIZE
from std.utils import IndexList
from layout import TensorLayout, TileTensor
from layout.swizzle import Swizzle
from layout.tensor_core import num_matrix_reg
from layout.tile_layout import Layout, row_major, col_major
from layout.tile_tensor import stack_allocation
from structured_kernels.amd_tile_io import (
    GMemTile,
    SMemTile,
    RegTile,
    load_lds_fragment,
)


# ===----------------------------------------------------------------------=== #
# QuadrantMmaOp
# ===----------------------------------------------------------------------=== #


struct QuadrantMmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    k_group_size: Int,
    num_k_groups: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    swizzle: Optional[Swizzle] = None,
]:
    """MMA operator for AMD matmul ping-pong schedule.

    Owns A/B/C register tiles in LOCAL address space. Provides quadrant
    load/compute methods for the ping-pong double-buffering schedule:
    load_a_quadrant/load_b_quadrant fill half the register tile from
    SMEM via load_lds_fragment, then mma_quadrant computes on it.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        k_group_size: Number of MMA K-groups per fragment load.
        num_k_groups: Number of k-groups across the full BK dimension.
        num_m_mmas: MMA tiles along M within the warp tile.
        num_n_mmas: MMA tiles along N within the warp tile.
        swizzle: Optional SMEM swizzle for load helpers.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime c_frag_size = num_matrix_reg[Self.MMA_M, Self.MMA_N]()

    # Fragment width for load_lds_fragment.
    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE

    # Derived SMEM warp tile shapes.
    comptime num_k_mmas = Self.num_k_groups * Self.k_group_size
    comptime WM = Self.num_m_mmas * Self.MMA_M
    comptime WN = Self.num_n_mmas * Self.MMA_N
    comptime BK = Self.num_k_mmas * Self.MMA_K

    # Quadrant shapes (ping-pong schedule).
    comptime quad_m = Self.num_m_mmas // 2
    comptime quad_n = Self.num_n_mmas // 2
    comptime quad_WM = Self.quad_m * Self.MMA_M
    comptime quad_WN = Self.quad_n * Self.MMA_N

    # Register layouts: [num_m_mmas, num_k_mmas * mma_frag_width]
    # so quadrant tiling works.
    comptime _a_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_k_mmas * Self._mma_frag_width,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas,
        Self.num_k_mmas * Self._mma_frag_width,
    ]()
    # C layout: [num_m_mmas, num_n_mmas * c_frag_size]
    # so quadrant tiling [quad_m, quad_n * c_frag](which_a, which_b) works.
    comptime accum_width = Self.c_frag_size
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas, Self.num_n_mmas * Self.accum_width
    ]()

    var _a_reg: TileTensor[
        Self.in_type,
        type_of(Self._a_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_reg: TileTensor[
        Self.in_type,
        type_of(Self._b_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _c_reg: TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        self._a_reg = stack_allocation[Self.in_type, address_space=.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[Self.in_type, address_space=.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[Self.out_type, address_space=.LOCAL](
            Self._c_reg_layout
        )
        comptime num_c_elems = (
            Self.num_m_mmas * Self.num_n_mmas * Self.c_frag_size
        )
        comptime for i in range(num_c_elems):
            self._c_reg.raw_store(i, Scalar[Self.out_type](0))

    # === Quadrant methods for ping-pong schedule ===
    # The schedule calls load_a_quadrant/load_b_quadrant to fill one half
    # of the register tile, then mma_quadrant to compute on it.

    @always_inline
    def load_a_quadrant[
        which: Int
    ](self, smem_tile: SMemTile[Self.in_type, _, _],):
        """Load A quadrant `which` from SMEM sub-tile to registers.

        Tiles a_reg as [quad_m, reg_cols](which, 0) to get the register
        sub-tile for this quadrant, then loads via load_lds_fragment.

        Parameters:
            which: Quadrant index selecting which half of the A register
                tile to fill.

        Args:
            smem_tile: SMEM sub-tile for this A quadrant of shape
                `quad_WM x BK`.
        """
        comptime assert type_of(smem_tile).static_shape[0] == Self.quad_WM
        comptime assert type_of(smem_tile).static_shape[1] == Self.BK

        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        var reg_quad = self._a_reg.tile[Self.quad_m, reg_cols](which, 0)
        load_lds_fragment[Self.MMA_K, Self.swizzle](smem_tile, reg_quad)

    @always_inline
    def load_b_quadrant[
        which: Int
    ](self, smem_tile: SMemTile[Self.in_type, _, _],):
        """Load B quadrant `which` from SMEM sub-tile to registers.

        Parameters:
            which: Quadrant index selecting which half of the B register
                tile to fill.

        Args:
            smem_tile: SMEM sub-tile for this B quadrant of shape
                `quad_WN x BK`.
        """
        comptime assert type_of(smem_tile).static_shape[0] == Self.quad_WN
        comptime assert type_of(smem_tile).static_shape[1] == Self.BK

        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        var reg_quad = self._b_reg.tile[Self.quad_n, reg_cols](which, 0)
        load_lds_fragment[Self.MMA_K, Self.swizzle](smem_tile, reg_quad)

    @always_inline
    def mma_quadrant[which_a: Int, which_b: Int](self):
        """Execute MMA for quadrant (which_a, which_b) via TiledMma.

        Slices A/B/C register tiles to the quadrant and delegates to
        TiledMma for stateless computation.

        Parameters:
            which_a: Quadrant index along M selecting the A and C
                row half.
            which_b: Quadrant index along N selecting the B and C
                column half.
        """
        comptime reg_cols = Self.num_k_mmas * Self._mma_frag_width
        comptime c_quad_cols = Self.quad_n * Self.c_frag_size

        var a_quad = self._a_reg.tile[Self.quad_m, reg_cols](which_a, 0)
        var b_quad = self._b_reg.tile[Self.quad_n, reg_cols](which_b, 0)
        var c_quad = self._c_reg.tile[Self.quad_m, c_quad_cols](
            which_a, which_b
        )
        TiledMma[Self.out_type, Self.in_type, Self.shape, Self.num_k_mmas].mma(
            a_quad, b_quad, c_quad
        )

    @always_inline
    def accum_tile(
        self,
    ) -> TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]:
        """Return the accumulator register tile."""
        return self._c_reg


# ===----------------------------------------------------------------------=== #
# TiledMma — stateless MMA computation on TileTensors
# ===----------------------------------------------------------------------=== #


struct TiledMma[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    group_size: Int,
]:
    """Stateless MMA computation on TileTensors.

    Direct TileTensor port of TiledTensorCore.mma. Iterates group_size
    k-steps, indexes A/B register tiles per step, and calls gpu_mma.
    No register ownership, no SMEM loading: pure computation.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        group_size: Number of k-steps per mma() call.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime a_frag_size = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    @staticmethod
    @always_inline
    def mma[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
    ](
        a_reg: TileTensor[Self.in_type, a_layout, _, address_space=.LOCAL],
        b_reg: TileTensor[Self.in_type, b_layout, _, address_space=.LOCAL],
        c_reg: TileTensor[
            Self.out_type, c_layout, MutUntrackedOrigin, address_space=.LOCAL
        ],
    ):
        """Execute group_size MMA operations across the K dimension.

        Mirrors TiledTensorCore.mma: iterates group_size k-steps, tiles
        A/B registers per step via vectorize, and accumulates into C.

        Parameters:
            a_layout: Inferred layout of A register tile.
            b_layout: Inferred layout of B register tile.
            c_layout: Inferred layout of C register tile.

        Args:
            a_reg: A fragments [num_m_mmas, group_size * a_frag_size].
            b_reg: B fragments [num_n_mmas, group_size * a_frag_size].
            c_reg: Accumulator [num_m_mmas, num_n_mmas * c_frag_size].
        """
        comptime a_frag_size = Self.a_frag_size
        comptime c_frag_size = Self.c_frag_size
        comptime num_m_mmas = a_layout.static_shape[0]
        comptime num_n_mmas = b_layout.static_shape[0]

        var a_vec = a_reg.vectorize[1, a_frag_size]()
        var b_vec = b_reg.vectorize[1, a_frag_size]()
        var c_vec = c_reg.vectorize[1, c_frag_size]()

        # Provide evidence for TileTensor.__getitem__'s `where flat_rank`
        # constraint. With inferred layouts the compiler can't prove rank
        # from the generic type alone — these asserts supply the proof.
        comptime assert type_of(a_vec).flat_rank == 2
        comptime assert type_of(b_vec).flat_rank == 2
        comptime assert type_of(c_vec).flat_rank == 2

        comptime for k in range(Self.group_size):
            comptime for m in range(num_m_mmas):
                comptime for n in range(num_n_mmas):
                    gpu_mma(
                        c_vec[m, n],
                        b_vec[n, k],
                        a_vec[m, k],
                        c_vec[m, n],
                    )


# ===----------------------------------------------------------------------=== #
# MmaOp — register ownership + SMEM loading + schedule API
# ===----------------------------------------------------------------------=== #


struct MmaOp[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    swizzle: Optional[Swizzle] = None,
]:
    """Register ownership + SMEM loading + schedule API for AMD matmul.

    Owns A/B/C register tiles in LOCAL address space. Provides the
    schedule-facing API: load_frag[k] loads from SMEM to registers,
    mma[k] delegates to TiledMma for computation.

    Parameters:
        out_type: Accumulator data type (typically float32).
        in_type: Input element data type (bfloat16 or float8).
        shape: MMA instruction shape [M, N, K].
        k_group_size: Number of MMA k-steps per fragment load.
        num_k_tiles: Number of k-tiles across the warp K dimension.
        num_m_mmas: MMA tiles along M within the warp tile.
        num_n_mmas: MMA tiles along N within the warp tile.
        swizzle: Optional SMEM swizzle for fragment loading.
    """

    comptime MMA_M = Self.shape[0]
    comptime MMA_N = Self.shape[1]
    comptime MMA_K = Self.shape[2]
    comptime c_frag_size = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    # simd_width = k_group_size * _mma_frag_width: elements per k-tile load.
    comptime simd_width = Self.k_group_size * Self._mma_frag_width

    # Derived SMEM warp tile shapes.
    comptime WM = Self.num_m_mmas * Self.MMA_M
    comptime WN = Self.num_n_mmas * Self.MMA_N
    comptime k_tile_size = Self.MMA_K * Self.k_group_size

    # Register layouts: height = num_mmas * num_k_tiles,
    # width = simd_width (one k-tile's worth of elements per row).
    comptime _a_reg_layout = row_major[
        Self.num_m_mmas * Self.num_k_tiles,
        Self.simd_width,
    ]()
    comptime _b_reg_layout = row_major[
        Self.num_n_mmas * Self.num_k_tiles,
        Self.simd_width,
    ]()
    comptime _c_reg_layout = row_major[
        Self.num_m_mmas,
        Self.num_n_mmas * Self.c_frag_size,
    ]()

    var _a_reg: TileTensor[
        Self.in_type,
        type_of(Self._a_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _b_reg: TileTensor[
        Self.in_type,
        type_of(Self._b_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    var _c_reg: TileTensor[
        Self.out_type,
        type_of(Self._c_reg_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    @always_inline
    def __init__(out self):
        self._a_reg = stack_allocation[Self.in_type, address_space=.LOCAL](
            Self._a_reg_layout
        )
        self._b_reg = stack_allocation[Self.in_type, address_space=.LOCAL](
            Self._b_reg_layout
        )
        self._c_reg = stack_allocation[Self.out_type, address_space=.LOCAL](
            Self._c_reg_layout
        )
        comptime num_c_elems = (
            Self.num_m_mmas * Self.num_n_mmas * Self.c_frag_size
        )
        comptime for i in range(num_c_elems):
            self._c_reg.raw_store(i, Scalar[Self.out_type](0))

    @always_inline
    def accum_tile(self) -> ref[self._c_reg] type_of(self._c_reg):
        return self._c_reg

    @always_inline
    def load_frag[
        k_tile_idx: Int
    ](
        self,
        a_smem_warp: SMemTile[Self.in_type, _, _],
        b_smem_warp: SMemTile[Self.in_type, _, _],
    ):
        """Load A and B MMA fragments for k-tile k_tile_idx from SMEM.

        Expects block-local warp tiles of shape WM x k_tile_size (or
        WN x k_tile_size), where each k-tile block is contiguous in
        SMEM (blocked_product layout). Uses direct distribute with
        swizzle: correct because each block starts at a
        swizzle-aligned offset.

        Parameters:
            k_tile_idx: Index of the k-tile to load into registers.

        Args:
            a_smem_warp: Block-local SMEM warp tile for A of shape
                `WM x k_tile_size`.
            b_smem_warp: Block-local SMEM warp tile for B of shape
                `WN x k_tile_size`.
        """
        comptime assert type_of(a_smem_warp).static_shape[0] == Self.WM
        comptime assert type_of(b_smem_warp).static_shape[0] == Self.WN
        comptime assert type_of(a_smem_warp).static_shape[1] == Self.k_tile_size
        comptime assert type_of(b_smem_warp).static_shape[1] == Self.k_tile_size

        # B fragments first for ds_read scheduling.
        comptime for i in range(Self.num_n_mmas):
            var b_row = k_tile_idx * Self.num_n_mmas + i
            var dist = (
                b_smem_warp.tile[Self.MMA_N, Self.k_tile_size](i, 0)
                .vectorize[1, Self.simd_width]()
                .distribute[
                    col_major[Self.MMA_N, WARP_SIZE // Self.MMA_N](),
                    swizzle=Self.swizzle,
                ](lane_id())
            )
            self._b_reg.vectorize[1, Self.simd_width]()[b_row, 0] = dist[0, 0]

        # A fragments.
        comptime for i in range(Self.num_m_mmas):
            var a_row = k_tile_idx * Self.num_m_mmas + i
            var dist = (
                a_smem_warp.tile[Self.MMA_M, Self.k_tile_size](i, 0)
                .vectorize[1, Self.simd_width]()
                .distribute[
                    col_major[Self.MMA_M, WARP_SIZE // Self.MMA_M](),
                    swizzle=Self.swizzle,
                ](lane_id())
            )
            self._a_reg.vectorize[1, Self.simd_width]()[a_row, 0] = dist[0, 0]

    @always_inline
    def mma[k_tile_idx: Int](self):
        """Execute MMA for k-tile k_tile_idx via TiledMma.

        Slices A/B registers for this k-tile and delegates to
        TiledMma.mma for stateless computation.

        Parameters:
            k_tile_idx: Index of the k-tile to compute MMA for.
        """
        var a_slice = self._a_reg.tile[Self.num_m_mmas, Self.simd_width](
            k_tile_idx, 0
        )
        var b_slice = self._b_reg.tile[Self.num_n_mmas, Self.simd_width](
            k_tile_idx, 0
        )
        TiledMma[
            Self.out_type, Self.in_type, Self.shape, Self.k_group_size
        ].mma(a_slice, b_slice, self._c_reg)
