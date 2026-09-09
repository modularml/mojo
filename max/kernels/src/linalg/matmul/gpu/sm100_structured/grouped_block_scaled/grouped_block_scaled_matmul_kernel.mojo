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
"""Grouped block-scaled SM100 matmul kernel for multiple GEMM problems.

This kernel extends the block_scaled_matmul_kernel to support grouped GEMM
with variable problem sizes per group. It uses:

1. GroupedTileScheduler: For linear tile iteration across groups
2. TMATensorTileArray: For per-block updatable TMA descriptors
3. Dynamic tensormap updates: When transitioning between groups

Architecture (aligned with NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
- TMA warp: Initializes A/B/SFA/SFB tensormaps, handles group transitions
- MMA warp: Consumes input tiles, performs block-scaled MMA
- Epilogue warps: Initializes C tensormap, handles C group transitions
- Named barrier synchronization between warps for tensormap init

Key differences from block_scaled_matmul_kernel.mojo:
1. TMA descriptors are per-block (TMATensorTileArray) not grid constants
2. SMEM tensormap buffers for dynamic updates (5 x 128 bytes)
3. GroupedWorkInfo provides group_idx, k_tile_count, group_changed
4. When group_changed=True, tensormaps are updated before loading tiles
5. K-loop uses per-group k_tile_count instead of global K dimension
"""

from std.collections import Optional
from std.math import ceildiv
from std.math.uutils import ufloordiv
from std.memory import UnsafePointer, Pointer
from std.sys import size_of

from max.gpu import WARP_SIZE, block_idx, lane_id
from max.gpu.memory import external_memory, fence_mbarrier_init
from max.gpu.primitives.cluster import cluster_sync, elect_one_sync
from max.gpu.sync import syncwarp
from max.gpu.host.nvidia.tma import TMADescriptor, TensorMapSwizzle
from std.sys import inlined_assembly
from layout import (
    Coord,
    ComptimeInt,
    Layout,
    RowMajorLayout,
    TileTensor,
    CoordLike,
)
from layout.tile_layout import _IntToComptimeInt
from structured_kernels.tile_types import (
    TmaOpType,
    tma_desc_layout_3d,
    tma_desc_layout_5d,
)
from layout.tma_async import (
    TMATensorTile,
    TMATensorTileArray,
)
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
    tile_sf_layout_k_major,
)

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
from linalg.fp4_utils import SF_MN_GROUP_SIZE, SF_ATOM_M, SF_ATOM_K
from linalg.utils import elementwise_compute_lambda_type
from ..structured_kernels.config import (
    BlockScaledMatmulConfig,
    OutputPipelineConfig,
)
from structured_kernels.kernel_common import (
    WarpRole,
    KernelContext,
    compute_tma_tile_dims,
    compute_accum_barrier_counts,
    compute_input_consumer_count,
    init_core_barriers,
    init_clc_barriers,
)
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    ProducerTiles,
    ConsumerTiles,
    OutputTilePipeline,
    BlockScaledTilePayload,
)
from ..structured_kernels.tmem import (
    BlockScaledTmem,
    TmemAllocation,
    TmemDeallocBarrier,
)
from structured_kernels.barriers import WarpGroupBarrier
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)
from ..structured_kernels.output_writer import TileWriter
from .grouped_block_scaled_smem import GroupedBlockScaledSmem
from .grouped_tile_scheduler import (
    GroupedTileScheduler,
    GroupedWorkInfo,
    GroupedCLCWorkIterator,
    GroupedCLCSchedulerIterator,
)


comptime _GroupPtrLayout[max_groups: Int] = RowMajorLayout[
    *Coord[ComptimeInt[max_groups], ComptimeInt[1]].element_types
]
comptime _GroupPtrTile[max_groups: Int] = TileTensor[
    .uint64, _GroupPtrLayout[max_groups], MutAnyOrigin
]
comptime _ProblemSizesLayout[max_groups: Int] = RowMajorLayout[
    *Coord[ComptimeInt[max_groups], ComptimeInt[4]].element_types
]
comptime _ProblemSizesTile[max_groups: Int] = TileTensor[
    .int32, _ProblemSizesLayout[max_groups], MutAnyOrigin
]


# =============================================================================
# Constants
# =============================================================================

# TMA descriptor size in bytes (128 bytes per descriptor)
comptime TMA_DESCRIPTOR_SIZE = 128

# Number of tensormaps for grouped GEMM (A, B, SFA, SFB, C)
comptime NUM_TENSORMAPS = 5


# =============================================================================
# GroupedTensormapSmem - SMEM storage for tensormap descriptors
# =============================================================================


@fieldwise_init
struct GroupedTensormapSmem(TrivialRegisterPassable):
    """Shared memory pointers for tensormap descriptors.

    Points to 5 TMA descriptors (128 bytes each) in SMEM for dynamic updates:
    - A, B, SFA, SFB for input loading
    - C for output storing

    These pointers should come from the main SMEM struct (GroupedBlockScaledSmem)
    to ensure all warps access the same SMEM locations.
    """

    @__allow_legacy_any_origin_fields
    var desc_a: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var desc_b: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var desc_sfa: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var desc_sfb: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space=.SHARED
    ]

    @__allow_legacy_any_origin_fields
    var desc_c: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space=.SHARED
    ]

    @staticmethod
    @always_inline
    def from_smem(
        ptr_a: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space=.SHARED
        ],
        ptr_b: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space=.SHARED
        ],
        ptr_sfa: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space=.SHARED
        ],
        ptr_sfb: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space=.SHARED
        ],
        ptr_c: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space=.SHARED
        ],
    ) -> Self:
        """Create tensormap pointers from explicit SMEM pointers.

        Args:
            ptr_a: Pointer to A tensormap in SMEM.
            ptr_b: Pointer to B tensormap in SMEM.
            ptr_sfa: Pointer to SFA tensormap in SMEM.
            ptr_sfb: Pointer to SFB tensormap in SMEM.
            ptr_c: Pointer to C tensormap in SMEM.

        Returns:
            GroupedTensormapSmem with the provided pointers.
        """
        return Self(ptr_a, ptr_b, ptr_sfa, ptr_sfb, ptr_c)


# =============================================================================
# GroupedTensormapManager - Manages tensormap updates for grouped GEMM
# =============================================================================


@fieldwise_init
struct GroupedTensormapManager(TrivialRegisterPassable):
    """Manages tensormap SMEM state and updates for grouped GEMM.

    Handles the 4-step CuTe DSL update pattern:
    1. tensormap_fence_acquire() - Acquire fence on block's GMEM tensormap
    2. replace_tensormap_global_address_in_shared_mem() - Update SMEM descriptor
    3. tensormap_cp_fence_release() - Copy SMEM -> block's GMEM tensormap
    4. syncwarp() - Sync before using updated tensormap

    TMA descriptor arrays are passed by reference (as Pointer from
    TMATensorTileArray[blk]) to methods rather than stored by value. This
    ensures PTX tensormap operations receive valid GMEM addresses with correct
    address space semantics.

    The manager stores only SMEM descriptor pointers, which are shared across
    all warps within a CTA.
    """

    # SMEM descriptors for in-place tensormap updates
    var smem: GroupedTensormapSmem

    @always_inline
    def init_ab_tensormaps[
        a_dtype: DType,
        a_rank: Int,
        a_tile_shape: IndexList[a_rank],
        a_desc_shape: IndexList[a_rank],
        b_dtype: DType,
        b_rank: Int,
        b_tile_shape: IndexList[b_rank],
        b_desc_shape: IndexList[b_rank],
        sfa_dtype: DType,
        sfa_rank: Int,
        sfa_tile_shape: IndexList[sfa_rank],
        sfa_desc_shape: IndexList[sfa_rank],
        sfb_dtype: DType,
        sfb_rank: Int,
        sfb_tile_shape: IndexList[sfb_rank],
        sfb_desc_shape: IndexList[sfb_rank],
    ](
        self,
        template_a: TMATensorTile[a_dtype, a_rank, a_tile_shape, a_desc_shape],
        template_b: TMATensorTile[b_dtype, b_rank, b_tile_shape, b_desc_shape],
        template_sfa: TMATensorTile[
            sfa_dtype, sfa_rank, sfa_tile_shape, sfa_desc_shape
        ],
        template_sfb: TMATensorTile[
            sfb_dtype, sfb_rank, sfb_tile_shape, sfb_desc_shape
        ],
    ):
        """Initialize A/B/SFA/SFB tensormaps in SMEM from grid-constant templates.

        Called by MMA warp (lane 0). Copies template descriptors to SMEM.
        Templates must be kernel parameters with nvvm.grid_constant metadata.

        Parameters:
            a_dtype: Element type of the A matrix tensor.
            a_rank: Tensor rank of the A matrix TMA descriptor.
            a_tile_shape: Per-tile shape of the A TMA load in each rank.
            a_desc_shape: Full tensor shape of the A matrix described by
                the TMA descriptor.
            b_dtype: Element type of the B matrix tensor.
            b_rank: Tensor rank of the B matrix TMA descriptor.
            b_tile_shape: Per-tile shape of the B TMA load in each rank.
            b_desc_shape: Full tensor shape of the B matrix described by
                the TMA descriptor.
            sfa_dtype: Element type of the A scale-factor tensor.
            sfa_rank: Tensor rank of the SFA TMA descriptor.
            sfa_tile_shape: Per-tile shape of the SFA TMA load in each
                rank.
            sfa_desc_shape: Full tensor shape of the SFA scale-factor
                tensor described by the TMA descriptor.
            sfb_dtype: Element type of the B scale-factor tensor.
            sfb_rank: Tensor rank of the SFB TMA descriptor.
            sfb_tile_shape: Per-tile shape of the SFB TMA load in each
                rank.
            sfb_desc_shape: Full tensor shape of the SFB scale-factor
                tensor described by the TMA descriptor.

        Args:
            template_a: Grid-constant TMA template whose descriptor is
                copied into the A SMEM tensormap slot.
            template_b: Grid-constant TMA template whose descriptor is
                copied into the B SMEM tensormap slot.
            template_sfa: Grid-constant TMA template whose descriptor is
                copied into the SFA SMEM tensormap slot.
            template_sfb: Grid-constant TMA template whose descriptor is
                copied into the SFB SMEM tensormap slot.
        """
        if lane_id() == 0:
            template_a.smem_tensormap_init(self.smem.desc_a)
            template_b.smem_tensormap_init(self.smem.desc_b)
            template_sfa.smem_tensormap_init(self.smem.desc_sfa)
            template_sfb.smem_tensormap_init(self.smem.desc_sfb)

    @always_inline
    def init_c_tensormap[
        c_dtype: DType,
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
    ](
        self,
        template_c: TMATensorTile[c_dtype, c_rank, c_tile_shape, c_desc_shape],
    ):
        """Initialize C tensormap in SMEM from grid-constant template.

        Called by epilogue warp (lane 0). Copies template descriptor to SMEM.

        Parameters:
            c_dtype: Element type of the C output matrix tensor.
            c_rank: Tensor rank of the C matrix TMA descriptor.
            c_tile_shape: Per-tile shape of the C TMA store in each rank.
            c_desc_shape: Full tensor shape of the C matrix described by
                the TMA descriptor.

        Args:
            template_c: Grid-constant TMA template whose descriptor is
                copied into the C SMEM tensormap slot.
        """
        if lane_id() == 0:
            template_c.smem_tensormap_init(self.smem.desc_c)

    @always_inline
    def update_ab_for_group[
        a_dtype: DType,
        a_rank: Int,
        a_tile_shape: IndexList[a_rank],
        a_desc_shape: IndexList[a_rank],
        b_dtype: DType,
        b_rank: Int,
        b_tile_shape: IndexList[b_rank],
        b_desc_shape: IndexList[b_rank],
        sfa_dtype: DType,
        sfa_rank: Int,
        sfa_tile_shape: IndexList[sfa_rank],
        sfa_desc_shape: IndexList[sfa_rank],
        sfb_dtype: DType,
        sfb_rank: Int,
        sfb_tile_shape: IndexList[sfb_rank],
        sfb_desc_shape: IndexList[sfb_rank],
        max_groups: Int,
    ](
        self,
        group_idx: UInt32,
        group_a_ptrs: _GroupPtrTile[max_groups],
        group_b_ptrs: _GroupPtrTile[max_groups],
        group_sfa_ptrs: _GroupPtrTile[max_groups],
        group_sfb_ptrs: _GroupPtrTile[max_groups],
        tma_a: UnsafePointer[
            TMATensorTile[a_dtype, a_rank, a_tile_shape, a_desc_shape],
            MutAnyOrigin,
        ],
        tma_b: UnsafePointer[
            TMATensorTile[b_dtype, b_rank, b_tile_shape, b_desc_shape],
            MutAnyOrigin,
        ],
        tma_sfa: UnsafePointer[
            TMATensorTile[sfa_dtype, sfa_rank, sfa_tile_shape, sfa_desc_shape],
            MutAnyOrigin,
        ],
        tma_sfb: UnsafePointer[
            TMATensorTile[sfb_dtype, sfb_rank, sfb_tile_shape, sfb_desc_shape],
            MutAnyOrigin,
        ],
    ):
        """Update A/B/SFA/SFB tensormaps for the specified group.

        Called when group_changed=True in TMA load warp.
        TMA pointers must be from TMATensorTileArray[block_idx.x] (GMEM).

        Parameters:
            a_dtype: Element type of the A matrix tensor.
            a_rank: Tensor rank of the A matrix TMA descriptor.
            a_tile_shape: Per-tile shape of the A TMA load in each rank.
            a_desc_shape: Full tensor shape of the A matrix described by
                the TMA descriptor.
            b_dtype: Element type of the B matrix tensor.
            b_rank: Tensor rank of the B matrix TMA descriptor.
            b_tile_shape: Per-tile shape of the B TMA load in each rank.
            b_desc_shape: Full tensor shape of the B matrix described by
                the TMA descriptor.
            sfa_dtype: Element type of the A scale-factor tensor.
            sfa_rank: Tensor rank of the SFA TMA descriptor.
            sfa_tile_shape: Per-tile shape of the SFA TMA load in each
                rank.
            sfa_desc_shape: Full tensor shape of the SFA scale-factor
                tensor described by the TMA descriptor.
            sfb_dtype: Element type of the B scale-factor tensor.
            sfb_rank: Tensor rank of the SFB TMA descriptor.
            sfb_tile_shape: Per-tile shape of the SFB TMA load in each
                rank.
            sfb_desc_shape: Full tensor shape of the SFB scale-factor
                tensor described by the TMA descriptor.
            max_groups: Maximum number of GEMM groups in the per-group
                pointer arrays.

        Args:
            group_idx: Index of the GEMM group whose tensor base
                addresses are loaded into the tensormaps.
            group_a_ptrs: Per-group array of GMEM base addresses for the
                A matrices.
            group_b_ptrs: Per-group array of GMEM base addresses for the
                B matrices.
            group_sfa_ptrs: Per-group array of GMEM base addresses for
                the SFA scale-factor tensors.
            group_sfb_ptrs: Per-group array of GMEM base addresses for
                the SFB scale-factor tensors.
            tma_a: Pointer to the per-block A tensormap in GMEM to
                update.
            tma_b: Pointer to the per-block B tensormap in GMEM to
                update.
            tma_sfa: Pointer to the per-block SFA tensormap in GMEM to
                update.
            tma_sfb: Pointer to the per-block SFB tensormap in GMEM to
                update.
        """
        # Step 1: Acquire fences on GMEM tensormaps
        tma_a[].tensormap_fence_acquire()
        tma_b[].tensormap_fence_acquire()
        tma_sfa[].tensormap_fence_acquire()
        tma_sfb[].tensormap_fence_acquire()

        # Step 2: Update SMEM descriptors (lane 0 only)
        if lane_id() == 0:
            var g = Int(group_idx)

            var a_ptr = UnsafePointer[mut=True, Scalar[a_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_a_ptrs[g, 0])
            )
            var b_ptr = UnsafePointer[mut=True, Scalar[b_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_b_ptrs[g, 0])
            )
            var sfa_ptr = UnsafePointer[
                mut=True, Scalar[sfa_dtype], MutAnyOrigin
            ](unsafe_from_address=Int(group_sfa_ptrs[g, 0]))
            var sfb_ptr = UnsafePointer[
                mut=True, Scalar[sfb_dtype], MutAnyOrigin
            ](unsafe_from_address=Int(group_sfb_ptrs[g, 0]))

            tma_a[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_a, a_ptr
            )
            tma_b[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_b, b_ptr
            )
            tma_sfa[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_sfa, sfa_ptr
            )
            tma_sfb[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_sfb, sfb_ptr
            )

        syncwarp()

        # Step 3: Fence release copies SMEM -> GMEM
        tma_a[].tensormap_cp_fence_release(self.smem.desc_a)
        tma_b[].tensormap_cp_fence_release(self.smem.desc_b)
        tma_sfa[].tensormap_cp_fence_release(self.smem.desc_sfa)
        tma_sfb[].tensormap_cp_fence_release(self.smem.desc_sfb)

        # Step 4: Sync within warp
        syncwarp()

    @always_inline
    def update_c_for_group[
        c_dtype: DType,
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        max_groups: Int,
    ](
        self,
        group_idx: UInt32,
        group_c_ptrs: _GroupPtrTile[max_groups],
        tma_c: UnsafePointer[
            TMATensorTile[c_dtype, c_rank, c_tile_shape, c_desc_shape],
            MutAnyOrigin,
        ],
    ):
        """Update C tensormap for the specified group.

        Called when group_changed=True in epilogue warp.
        TMA pointer must be from TMATensorTileArray[block_idx.x] (GMEM).

        Parameters:
            c_dtype: Element type of the C output matrix tensor.
            c_rank: Tensor rank of the C matrix TMA descriptor.
            c_tile_shape: Per-tile shape of the C TMA store in each rank.
            c_desc_shape: Full tensor shape of the C matrix described by
                the TMA descriptor.
            max_groups: Maximum number of GEMM groups in the per-group
                pointer arrays.

        Args:
            group_idx: Index of the GEMM group whose C tensor base
                address is loaded into the tensormap.
            group_c_ptrs: Per-group array of GMEM base addresses for the
                C output matrices.
            tma_c: Pointer to the per-block C tensormap in GMEM to update.
        """
        # Step 1: Acquire fence
        tma_c[].tensormap_fence_acquire()

        # Step 2: Update SMEM descriptor (lane 0 only)
        if lane_id() == 0:
            var g = Int(group_idx)
            var c_ptr = UnsafePointer[mut=True, Scalar[c_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_c_ptrs[g, 0])
            )

            tma_c[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_c, c_ptr
            )

        syncwarp()

        # Step 3: Fence release
        tma_c[].tensormap_cp_fence_release(self.smem.desc_c)

        # Step 4: Sync within warp
        syncwarp()


# =============================================================================
# Validation Utilities (matching NVIDIA CuTe DSL constraints)
# =============================================================================


def is_valid_dtypes_and_scale_factor_vec_size(
    ab_dtype: DType,
    sf_dtype: DType,
    sf_vec_size: Int,
    c_dtype: DType,
) -> Bool:
    """Check if dtypes and sf_vec_size are valid combinations.

    Valid combinations (from NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
    - MXF8: Float8E5M2/Float8E4M3FN + Float8E8M0FNU + sf_vec_size=32
    - MXF4: Float4E2M1FN + Float8E8M0FNU + sf_vec_size=32
    - NVF4: Float4E2M1FN + Float8E8M0FNU/Float8E4M3FN + sf_vec_size=16

    Args:
        ab_dtype: The data type of A and B matrices.
        sf_dtype: The data type of scale factors.
        sf_vec_size: The vector size of scale factors (16 or 32).
        c_dtype: The data type of the output matrix.

    Returns:
        True if the combination is valid.
    """
    # Check valid ab_dtype (FP8 or FP4 types)
    var valid_ab = ab_dtype in (
        DType.float4_e2m1fn,  # NVF4, MXF4
        DType.float8_e5m2,  # MXF8
        DType.float8_e4m3fn,  # MXF8
    )
    if not valid_ab:
        return False

    # Check valid sf_vec_size (16 for NVF4, 32 for MXF8/MXF4)
    if sf_vec_size not in (16, 32):
        return False

    # Check valid sf_dtype
    var valid_sf = sf_dtype in (
        DType.float8_e8m0fnu,  # MXF8, MXF4, NVF4
        DType.float8_e4m3fn,  # NVF4 only
    )
    if not valid_sf:
        return False

    # Check sf_dtype and sf_vec_size combinations
    # Float8E4M3FN scale factors only valid with sf_vec_size=16 (NVF4)
    if sf_dtype == .float8_e4m3fn and sf_vec_size == 32:
        return False

    # MXF8 (Float8) requires sf_vec_size=32
    if (
        ab_dtype in (DType.float8_e5m2, DType.float8_e4m3fn)
        and sf_vec_size == 16
    ):
        return False

    # Check valid c_dtype
    var valid_c = c_dtype in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float8_e5m2,
        DType.float8_e4m3fn,
    )
    if not valid_c:
        return False

    return True


def is_valid_mma_tiler_and_cluster_shape(
    mma_tiler_m: Int,
    mma_tiler_n: Int,
    cluster_m: Int,
    cluster_n: Int,
) -> Bool:
    """Check if MMA tiler and cluster shape are valid.

    Constraints (from NVIDIA CuTe DSL):
    - MMA tiler M: 128 or 256
    - MMA tiler N: 128 or 256
    - Cluster M must be multiple of 2 if MMA tiler M is 256
    - Cluster M/N: Power of 2, <=4 per axis (for SF multicast)
    - Total cluster size: <=16

    Args:
        mma_tiler_m: MMA tile height in the M dimension; must be 128
            or 256.
        mma_tiler_n: MMA tile width in the N dimension; must be 128
            or 256.
        cluster_m: Number of CTAs in the cluster along the M dimension;
            must be a power of 2 and at most 4, and a multiple of 2 when
            `mma_tiler_m` is 256.
        cluster_n: Number of CTAs in the cluster along the N dimension;
            must be a power of 2 and at most 4.

    Returns:
        True if the combination is valid.
    """
    # Check MMA tiler
    if mma_tiler_m not in (128, 256):
        return False
    if mma_tiler_n not in (128, 256):
        return False

    # Check cluster constraints
    if mma_tiler_m == 256 and cluster_m % 2 != 0:
        return False

    def is_power_of_2(x: Int) -> Bool:
        return x > 0 and (x & (x - 1)) == 0

    if not is_power_of_2(cluster_m) or not is_power_of_2(cluster_n):
        return False

    # SF multicast constraint
    if cluster_m > 4 or cluster_n > 4:
        return False

    # Total cluster size
    if cluster_m * cluster_n > 16:
        return False

    return True


# =============================================================================
# GroupedBlockScaledMatmulKernel - Main kernel struct
# =============================================================================


struct GroupedBlockScaledMatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    # Configuration
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Grouped GEMM parameters
    max_groups: Int,
    # Cluster shape (for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Epilogue fusion parameters
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
]:
    """Grouped block-scaled matmul kernel with dynamic tensormap updates.

    This kernel extends BlackwellBlockScaledMatmulKernel to support grouped GEMM:
    - Uses GroupedTileScheduler for linear tile iteration across groups
    - Uses GroupedTensormapManager for per-block updatable TMA descriptors
    - Updates tensormaps when transitioning between groups

    Architecture (aligned with NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
    - TMA warp: Initializes A/B/SFA/SFB tensormaps, handles group transitions
    - MMA warp: Waits for tensormap init, consumes tiles, performs block-scaled MMA
    - Epilogue warps: Initializes C tensormap, handles C group transitions

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        sfa_dtype: Element type of the A block scale factors.
        sfb_dtype: Element type of the B block scale factors.
        transpose_b: Whether B is stored transposed; must be `True`.
        config: Tile shapes, pipeline stages, and swizzle configuration.
        max_groups: Maximum number of GEMM groups supported at runtime.
        cluster_shape: CTA cluster dimensions as
            `(cluster_m, cluster_n, cluster_k)` (defaults to `(1, 1, 1)`).
        elementwise_compute_lambda_fn: Optional epilogue fusion
            lambda applied to accumulated results (defaults to `None`).
    """

    # ========== Derived Constants (from config) ==========

    comptime register_based_epilogue = Self.config.register_based_epilogue

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime accum_type = DType.float32  # Hardcoded for block-scaled
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE

    comptime NUM_THREADS = (
        Self.SCHEDULER_THREADS
        + Self.TMA_LOAD_THREADS
        + Self.MMA_THREADS
        + Self.EPILOGUE_THREADS
    )

    # ========== Tensormap Synchronization ==========

    # Named barrier for TMA->MMA tensormap init synchronization
    comptime TENSORMAP_AB_INIT_BARRIER_ID: Int = 3
    comptime TENSORMAP_AB_INIT_THREADS: Int = Self.TMA_LOAD_THREADS + Self.MMA_THREADS

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Self.config.k_group_size
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration — stride matches MMA output width for scaled kernels.
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (Self.MMA_N // 32)
    comptime stage_stride_cols = Self.MMA_N

    # Output pipeline config (bundles accum stages, stride, and cta_group)
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Barrier Arrival Counts ==========

    comptime _accum_barrier_counts = compute_accum_barrier_counts[
        Self.EPILOGUE_THREADS, Self.cta_group
    ]()
    comptime accum_pipeline_producer_arv_count = Self._accum_barrier_counts[0]
    comptime accum_pipeline_consumer_arv_count = Self._accum_barrier_counts[1]

    # ========== CLC Configuration for 2SM ==========
    # These are used by run_2sm() for CLC-based work distribution

    comptime num_clc_pipeline_stages_2sm: Int = 2  # Use 2 stages for 2SM
    comptime clc_producer_arv_count = 1  # Scheduler warp produces work
    # Only MMA warp calls wait_and_advance() which signals empty barrier.
    # Each thread in MMA warp on all CTAs calls arrive_cluster(0).
    comptime clc_consumer_arv_count = Self.MMA_THREADS * Self.cta_group
    comptime clc_throttle_producer_arv_count = Self.TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = Self.SCHEDULER_THREADS

    # ========== Grouped Tensormap Manager Type ==========
    # The manager stores only SMEM state. TMA descriptor references are
    # passed to methods with explicit GLOBAL address space.

    comptime TensormapManagerType = GroupedTensormapManager

    # ========== Grouped Tile Scheduler Type ==========

    comptime SchedulerType = GroupedTileScheduler[
        tile_m=Self.BM,
        tile_n=Self.BN,
        tile_k=Self.BK,
        max_groups=Self.max_groups,
    ]

    # ========== TMA Descriptor Array Types ==========
    # Per-block updatable tensormaps (not grid constants)

    comptime TMATensorTileArrayA = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.a_type,
        Self.ATmaOp.rank,
        Self.ATmaOp.tile_shape,
        Self.ATmaOp.desc_shape,
    ]
    comptime TMATensorTileArrayB = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.b_type,
        Self.BTmaOp.rank,
        Self.BTmaOp.tile_shape,
        Self.BTmaOp.desc_shape,
    ]
    comptime TMATensorTileArraySFA = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.sfa_dtype,
        Self.SFATmaOp.rank,
        Self.SFATmaOp.tile_shape,
        Self.SFATmaOp.desc_shape,
    ]
    comptime TMATensorTileArraySFB = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.sfb_dtype,
        Self.SFBTmaOp.rank,
        Self.SFBTmaOp.tile_shape,
        Self.SFBTmaOp.desc_shape,
    ]
    comptime TMATensorTileArrayC = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.c_type,
        Self.CTmaOp.rank,
        Self.CTmaOp.tile_shape,
        Self.CTmaOp.desc_shape,
    ]

    # ========== Per-Group Pointer Layout ==========
    # Layout for arrays of per-group tensor pointers

    comptime GroupPtrLayout = _GroupPtrLayout[Self.max_groups]
    comptime GroupPtrTile = _GroupPtrTile[Self.max_groups]

    # ========== Shared Memory Layout Types ==========

    comptime a_smem_layout = tile_layout_k_major_typed[
        Self.a_type, Self.BM, Self.BK, swizzle_mode=Self.config.a_swizzle
    ].to_layout()

    comptime b_smem_layout = tile_layout_k_major_typed[
        Self.b_type, Self.BN, Self.BK, swizzle_mode=Self.config.b_swizzle
    ].to_layout() if Self.transpose_b else tile_layout_mn_major_typed[
        Self.b_type, Self.BN, Self.BK, swizzle_mode=Self.config.b_swizzle
    ].to_layout()

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size
    comptime SF_K_GROUP_SIZE = SF_ATOM_K * Self.config.vec_sf_size

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        Self.MMA_N,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    # ========== Shared Memory Type ==========
    # Use GroupedBlockScaledSmem which includes SMEM storage for TMA descriptors.
    # This allows runtime updates to TMA base pointers for multi-group support.

    comptime SmemType = GroupedBlockScaledSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== MMA Operation Type ==========

    comptime MmaOp = MmaOpSM100_BlockScaled_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.config.scaling_kind,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=Self.transpose_b,
    ]

    # ========== Kernel Context Type ==========

    comptime Context = KernelContext[
        0,  # num_clc_pipeline_stages = 0 for grouped (no CLC)
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

    # ========== Tile Pipeline Types ==========
    # TileTensor-native payload - passed directly to TMA/MMA

    comptime TilePayload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        IndexList[2](
            Self.SmemType.Core.BM, Self.SmemType.Core.BK
        ),  # A tile shape
        IndexList[2](
            Self.SmemType.Core.BN, Self.SmemType.Core.BK
        ),  # B tile shape
        IndexList[2](
            Self.SmemType.Core.SFA_DIM0, Self.SmemType.Core.SFA_DIM1
        ),  # SFA shape
        IndexList[2](
            Self.SmemType.Core.SFB_DIM0, Self.SmemType.Core.SFB_DIM1
        ),  # SFB shape
        Self.SmemType.Core.num_pipeline_stages,
    ]

    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.Core.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== TMEM and Output Pipeline Types ==========

    comptime Tmem = TmemAllocation[Self.opc.cta_group]

    comptime TmemRegion = BlockScaledTmem[
        Self.accum_type,
        Self.MMA_M,
        Self.MMA_N,
        Self.num_accum_pipeline_stages,
        Self.sfa_dtype,
        Self.BM,
        Self.num_pipeline_stages,
        cta_group=Self.cta_group,
        num_sf_k_tiles=Self.config.num_sf_k_tiles,
    ]

    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    # ========== Warp Context Types ==========

    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]

    # Barrier for tensormap init synchronization between MMA and TMA warps.
    # MMA warp initializes SMEM tensormaps, TMA warp waits before using them.
    # Following CuTe DSL pattern: tensormap_ab_init_barrier (64 threads)
    comptime TensormapAbInitBarrier = WarpGroupBarrier[
        Self.TENSORMAP_AB_INIT_THREADS, Self.TENSORMAP_AB_INIT_BARRIER_ID
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Tile Writer Type ==========

    comptime TileWriterType = TileWriter[
        a_type=Self.a_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=Self.config.AB_swapped,
        c_smem_dim0=Self.SmemType.Core.OutputM,
        c_smem_dim1=Self.SmemType.Core.OutputN,
        num_output_stages=Self.config.num_output_stages,
        num_output_warps=Self.num_output_warps,
        elementwise_compute_lambda_fn=Self.elementwise_compute_lambda_fn,
        register_based_epilogue=Self.register_based_epilogue,
        batched=True,
    ]

    # ========== TMA Load Size Constants ==========

    comptime a_expected_bytes = Self.BM * Self.BK * size_of[Self.a_type]()
    comptime b_expected_bytes = Self.BN * Self.BK * size_of[Self.b_type]()
    comptime sfa_expected_bytes = Self.sfa_smem_layout.size() * size_of[
        Self.sfa_dtype
    ]()
    comptime sfb_expected_bytes = Self.sfb_smem_layout.size() * size_of[
        Self.sfb_dtype
    ]()

    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.sfa_expected_bytes
        + Self.sfb_expected_bytes
    ) * Self.config.k_group_size

    # ========== TMA Layouts (computed from config, new Layout types) ==========
    # 3D batched layouts for A, B, C (batch dim = 1 for per-group updates)

    comptime _tma_tile_dims = compute_tma_tile_dims[
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.OutputM,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
        Self.cta_group,
        AB_swapped=Self.config.AB_swapped,
    ]()
    comptime a_tile_dim0 = Self._tma_tile_dims[0]
    comptime b_tile_dim0 = Self._tma_tile_dims[1]
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()
    comptime c_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.c_type
    ]()

    # C tile dims -- same AB_swapped-aware logic as other kernels
    comptime c_tile_dim0 = Self._tma_tile_dims[2]
    comptime c_tile_dim1 = Self.c_swizzle_elems if (
        Self.config.AB_swapped
    ) else Self.OutputN

    # A, B, C: 3D TMA layouts (batch=1, rows, cols)
    comptime ATileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.a_tile_dim0, Self.BK]
    ]
    comptime ADescLayout = tma_desc_layout_3d[
        Self.a_type, 1, Self.a_tile_dim0, Self.config.a_swizzle
    ]
    comptime BTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.b_tile_dim0, Self.BK]
    ]
    comptime BDescLayout = tma_desc_layout_3d[
        Self.b_type, 1, Self.b_tile_dim0, Self.config.b_swizzle
    ]
    comptime CTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.c_tile_dim0, Self.c_tile_dim1]
    ]
    comptime CDescLayout = tma_desc_layout_3d[
        Self.c_type, 1, Self.c_tile_dim0, Self.config.c_swizzle
    ]

    # SFA, SFB: 5D TMA layouts (batch=1, then 4D scale factor dims)
    comptime SFATileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.BM // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFADescLayout = tma_desc_layout_5d[
        Self.sfa_dtype,
        1,
        Self.BM // SF_MN_GROUP_SIZE,
        Self.config.num_sf_k_tiles,
        SF_ATOM_M[0],
        TensorMapSwizzle.SWIZZLE_NONE,
    ]
    comptime SFBTileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.MMA_N // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFBDescLayout = tma_desc_layout_5d[
        Self.sfb_dtype,
        1,
        Self.MMA_N // SF_MN_GROUP_SIZE,
        Self.config.num_sf_k_tiles,
        SF_ATOM_M[0],
        TensorMapSwizzle.SWIZZLE_NONE,
    ]

    # TMA operation types
    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime CTmaOp = TmaOpType[Self.c_type, Self.CTileLayout, Self.CDescLayout]
    comptime SFATmaOp = TmaOpType[
        Self.sfa_dtype, Self.SFATileLayout, Self.SFADescLayout
    ]
    comptime SFBTmaOp = TmaOpType[
        Self.sfb_dtype, Self.SFBTileLayout, Self.SFBDescLayout
    ]

    # TMA load size constants
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Validation ==========

    @staticmethod
    def validate_config():
        """Compile-time validation of kernel configuration."""
        comptime assert (
            Self.a_type == Self.b_type
        ), "A and B types must match for block-scaled GEMM"
        comptime assert (
            Self.sfa_dtype == Self.sfb_dtype
        ), "SFA and SFB types must match"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "Only support cta_group == 1 or 2"
        comptime assert Self.max_groups >= 1, "max_groups must be at least 1"
        comptime assert Self.transpose_b, "Only support transposed B"

    # ========== TMA Update Helper ==========

    @staticmethod
    @always_inline
    def _update_tensormap_address[
        dtype: DType,
        tma_rank: Int,
        cta_tile_shape: IndexList[tma_rank],
        desc_shape: IndexList[tma_rank],
    ](
        tma_desc_ptr: UnsafePointer[
            TMATensorTile[dtype, tma_rank, cta_tile_shape, desc_shape],
            MutAnyOrigin,
        ],
        new_addr: Int,
    ):
        """Update TMA descriptor global address using inline assembly.

        Args:
            tma_desc_ptr: Pointer to the TMA descriptor in device memory.
            new_addr: New global memory address for the tensor.
        """
        # Get pointer to the descriptor bytes
        var desc_byte_ptr = tma_desc_ptr.bitcast[NoneType]()

        # Use PTX instruction to update the global address
        inlined_assembly[
            "tensormap.replace.tile.global_address.global.b1024.b64 [$0], $1;",
            NoneType,
            constraints="l,l",
            has_side_effect=True,
        ](desc_byte_ptr, new_addr)

    # ========== Kernel Entry Point ==========

    # ========== Problem Sizes Layout ==========
    # Layout for problem_sizes tensor: (max_groups, 4) with [M, N, K, L] per group
    comptime ProblemSizesLayout = _ProblemSizesLayout[Self.max_groups]
    comptime ProblemSizesTile = _ProblemSizesTile[Self.max_groups]

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    def init_barriers(
        ctx: Self.Context,
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors (1SM path, no CLC).

        Args:
            ctx: Kernel context with cluster and CTA election state.
            a_tma_template: Grid-constant TMA template for A loads.
            b_tma_template: Grid-constant TMA template for B loads.
            c_tma_template: Grid-constant TMA template for C stores.
            sfa_tma_template: Grid-constant TMA template for SFA loads.
            sfb_tma_template: Grid-constant TMA template for SFB loads.
            input_barriers: SMEM mbarrier array for input pipeline
                synchronization.
            accum_barriers: SMEM mbarrier array for accumulator pipeline
                synchronization.
            tmem_dealloc: SMEM barrier for TMEM deallocation
                synchronization.
        """
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_template.prefetch_descriptor()
            b_tma_template.prefetch_descriptor()
            c_tma_template.prefetch_descriptor()
            sfa_tma_template.prefetch_descriptor()
            sfb_tma_template.prefetch_descriptor()

            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M, Self.CLUSTER_N, Self.cta_group
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(Self.EPILOGUE_THREADS * Self.cta_group),
            )

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    def init_barriers_2sm(
        ctx: Self.Context,
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        clc_throttle: Self.SmemType.Pipelines.ClcThrottleBarriers,
        clc_full: Self.SmemType.Pipelines.ClcBarriers,
        clc_empty: Self.SmemType.Pipelines.ClcBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors (2SM path, with CLC).

        Args:
            ctx: Kernel context with cluster and CTA elected state.
            a_tma_template: Grid-constant TMA template for A loads.
            b_tma_template: Grid-constant TMA template for B loads.
            c_tma_template: Grid-constant TMA template for C stores.
            sfa_tma_template: Grid-constant TMA template for SFA loads.
            sfb_tma_template: Grid-constant TMA template for SFB loads.
            input_barriers: SMEM mbarrier array for input pipeline
                synchronization.
            accum_barriers: SMEM mbarrier array for accumulator pipeline
                synchronization.
            clc_throttle: SMEM throttle barriers limiting in-flight CLC
                work between scheduler and consumer warps.
            clc_full: SMEM mbarrier array signalling CLC buffer slots
                are full (producer to consumer).
            clc_empty: SMEM mbarrier array signalling CLC buffer slots
                are empty (consumer to producer).
            tmem_dealloc: SMEM barrier for TMEM deallocation
                synchronization.
        """
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_template.prefetch_descriptor()
            b_tma_template.prefetch_descriptor()
            c_tma_template.prefetch_descriptor()
            sfa_tma_template.prefetch_descriptor()
            sfb_tma_template.prefetch_descriptor()

            # cta_group=2 for 2SM path
            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M, Self.CLUSTER_N, 2
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(Self.EPILOGUE_THREADS * 2),
            )

            init_clc_barriers[Self.num_clc_pipeline_stages_2sm](
                clc_full.ptr,
                clc_empty.ptr,
                Self.clc_producer_arv_count,
                Int32(Self.clc_consumer_arv_count),
            )

            # Custom throttle pattern for 2SM grouped kernel
            comptime for i in range(Self.num_clc_pipeline_stages_2sm * 2):
                clc_throttle.ptr[i].init(
                    Int32(
                        Self.clc_throttle_producer_arv_count if i
                        < Self.num_clc_pipeline_stages_2sm else Self.clc_throttle_consumer_arv_count
                    )
                )

        fence_mbarrier_init()
        cluster_sync()

    # ========== 1SM Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_template, `nvvm.grid_constant`)
    @__name(
        StaticString(Self.config.get_kernel_name())
        + StaticString(
            "_fused_compute_epi" if Self.elementwise_compute_lambda_fn
            is not None else ""
        ),
    )
    def run(
        # Template tensormaps for SMEM initialization
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        # Per-block updatable tensormaps
        device_tma_a: Self.TMATensorTileArrayA,
        device_tma_b: Self.TMATensorTileArrayB,
        device_tma_sfa: Self.TMATensorTileArraySFA,
        device_tma_sfb: Self.TMATensorTileArraySFB,
        device_tma_c: Self.TMATensorTileArrayC,
        # Per-group pointer arrays (uint64 addresses)
        group_a_ptrs_lt: Self.GroupPtrTile,
        group_b_ptrs_lt: Self.GroupPtrTile,
        group_c_ptrs_lt: Self.GroupPtrTile,
        group_sfa_ptrs_lt: Self.GroupPtrTile,
        group_sfb_ptrs_lt: Self.GroupPtrTile,
        # Per-group problem sizes: (num_groups, 4) with [M, N, K, L]
        problem_sizes_lt: Self.ProblemSizesTile,
        # Number of active groups
        num_groups: Int32,
    ):
        """Grouped block-scaled GEMM kernel entry point.

        This kernel processes multiple GEMM problems (groups) with dynamic
        tensormap updates at group boundaries.

        Args:
            a_tma_template: Grid-constant TMA template for A loads, used
                to initialize the per-block A tensormap in SMEM.
            b_tma_template: Grid-constant TMA template for B loads, used
                to initialize the per-block B tensormap in SMEM.
            c_tma_template: Grid-constant TMA template for C stores, used
                to initialize the per-block C tensormap in SMEM.
            sfa_tma_template: Grid-constant TMA template for SFA loads,
                used to initialize the per-block SFA tensormap in SMEM.
            sfb_tma_template: Grid-constant TMA template for SFB loads,
                used to initialize the per-block SFB tensormap in SMEM.
            device_tma_a: Per-block updatable TMA descriptor array for A,
                indexed by `block_idx.x`.
            device_tma_b: Per-block updatable TMA descriptor array for B,
                indexed by `block_idx.x`.
            device_tma_sfa: Per-block updatable TMA descriptor array for
                SFA, indexed by `block_idx.x`.
            device_tma_sfb: Per-block updatable TMA descriptor array for
                SFB, indexed by `block_idx.x`.
            device_tma_c: Per-block updatable TMA descriptor array for C,
                indexed by `block_idx.x`.
            group_a_ptrs_lt: Per-group array of GMEM base addresses for
                the A matrices.
            group_b_ptrs_lt: Per-group array of GMEM base addresses for
                the B matrices.
            group_c_ptrs_lt: Per-group array of GMEM base addresses for
                the C matrices.
            group_sfa_ptrs_lt: Per-group array of GMEM base addresses for
                the SFA scale-factor tensors.
            group_sfb_ptrs_lt: Per-group array of GMEM base addresses for
                the SFB scale-factor tensors.
            problem_sizes_lt: Per-group problem sizes as a
                `(max_groups, 4)` tensor with `[M, N, K, L]` per group.
            num_groups: Number of active GEMM groups to process.
        """
        var _num_groups = Int(num_groups)
        Self.validate_config()

        # Alias kernel args for internal methods
        var group_a_ptrs = group_a_ptrs_lt
        var group_b_ptrs = group_b_ptrs_lt
        var group_c_ptrs = group_c_ptrs_lt
        var group_sfa_ptrs = group_sfa_ptrs_lt
        var group_sfb_ptrs = group_sfb_ptrs_lt
        var problem_sizes = problem_sizes_lt

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM accessors
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Kernel Context =====
        var ctx = Self.Context(tmem_addr_storage)

        # ===== Grouped Tile Scheduler =====
        var scheduler = Self.SchedulerType(problem_sizes, _num_groups)

        # ===== Barrier Initialization =====
        Self.init_barriers(
            ctx,
            a_tma_template,
            b_tma_template,
            c_tma_template,
            sfa_tma_template,
            sfb_tma_template,
            input_barriers,
            accum_barriers,
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            var load_iter = scheduler.work_iterator()
            var blk = block_idx.x
            var tensormap_init_done = False

            # Tensormap manager for SMEM descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )

            with input_pipeline.producer() as producer:
                for current in load_iter:
                    # Wait for MMA warp to init tensormaps (first time only)
                    if not tensormap_init_done:
                        Self.TensormapAbInitBarrier.sync()
                        tensormap_init_done = True

                    # === LOOKAHEAD PATTERN (CuteDSL style) ===
                    # Initialize to "ready" (True), only peek if there's work.
                    # This avoids wasted try_acquire when num_k_iters == 0.
                    var num_k_iters = Int(current.k_tile_count)
                    var next_ready = True
                    if num_k_iters > 0:
                        next_ready = producer.try_acquire()

                    # Update tensormaps on group change (overlaps with peek)
                    if current.group_changed:
                        tensormap_mgr.update_ab_for_group(
                            current.group_idx,
                            group_a_ptrs,
                            group_b_ptrs,
                            group_sfa_ptrs,
                            group_sfb_ptrs,
                            device_tma_a[blk],
                            device_tma_b[blk],
                            device_tma_sfa[blk],
                            device_tma_sfb[blk],
                        )

                    # Load tiles using lookahead pattern
                    for k_tile in range(num_k_iters):
                        with producer.acquire_if_needed(next_ready) as tiles:
                            Self.load_input_tiles(
                                device_tma_a[blk][],
                                device_tma_b[blk][],
                                device_tma_sfa[blk][],
                                device_tma_sfb[blk][],
                                tiles,
                                ctx.peer_cta_coord,
                                (
                                    Int(current.m),
                                    Int(current.n),
                                    0,  # batch = 0 for grouped
                                ),
                                ctx.a_multicast_mask,
                                ctx.b_multicast_mask,
                                UInt32(k_tile),
                                ctx.elect_one_cta,
                            )
                        # Peek for next iteration (CuteDSL style):
                        # Reset to ready, then conditionally peek.
                        next_ready = True
                        if k_tile + 1 < num_k_iters:
                            next_ready = producer.try_acquire()

                    syncwarp()

                producer.drain()

        # ===== SCHEDULER WARP =====
        # For grouped GEMM, no CLC scheduling is needed (num_clc_pipeline_stages=0).
        # The scheduler warp just returns immediately, matching working kernel pattern.
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # No CLC for grouped GEMM - just return
            return

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            var mma_iter = scheduler.work_iterator()

            # Initialize SMEM tensormaps from templates (MMA warp, per CuTe DSL)
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )
            tensormap_mgr.init_ab_tensormaps(
                a_tma_template,
                b_tma_template,
                sfa_tma_template,
                sfb_tma_template,
            )

            # Barrier sync with TMA warp - signal init complete
            Self.TensormapAbInitBarrier.sync()

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            with mma_ctx:
                for current in mma_iter:
                    if ctx.elect_one_cta:
                        with mma_ctx.output_pipeline.producer() as output_stage:
                            var tmem_offset = UInt32(output_stage.tmem.offset())

                            with input_pipeline.consumer() as consumer:
                                var num_k_iters = Int(current.k_tile_count)

                                # === LOOKAHEAD PATTERN (CuteDSL style) ===
                                # Initialize to "ready" (True), only peek if there's work.
                                var next_ready = True
                                if num_k_iters > 0:
                                    next_ready = consumer.try_acquire()

                                for k_tile in range(num_k_iters):
                                    with consumer.acquire_if_needed(
                                        next_ready
                                    ) as input_tiles:
                                        Self.mma(
                                            input_tiles,
                                            mma_op,
                                            tmem_offset,
                                            tmem_region,
                                            UInt32(k_tile),
                                            0,  # k_start = 0 for each group
                                        )
                                    # Peek for next iteration (CuteDSL style):
                                    # Reset to ready, then conditionally peek.
                                    next_ready = True
                                    if k_tile + 1 < num_k_iters:
                                        next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            var blk = block_idx.x

            # Tensormap manager for C descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )

            # Init C tensormap in SMEM (epilogue warp, per CuTe DSL)
            tensormap_mgr.init_c_tensormap(c_tma_template)
            syncwarp()

            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var epi_iter = scheduler.work_iterator()

            with epi_ctx:
                for current in epi_iter:
                    # Update C tensormap on group change
                    if current.group_changed:
                        tensormap_mgr.update_c_for_group(
                            current.group_idx,
                            group_c_ptrs,
                            device_tma_c[blk],
                        )

                    # Get current group's M, N dimensions
                    var g = Int(current.group_idx)
                    var group_m = UInt32(Int(problem_sizes[g, 0]))
                    var group_n = UInt32(Int(problem_sizes[g, 1]))

                    # Use per-block GMEM tensormap for epilogue
                    with epi_ctx.output_pipeline.consumer() as output_stage:
                        Self.epilogue(
                            c_tiles,
                            device_tma_c[blk][],
                            output_stage,
                            (current.m, current.n, UInt32(0)),
                            group_m,
                            group_n,
                        )

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    def load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_tile_coord: Tuple[Int, Int, Int],
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A, B, SFA, SFB tiles using TMA with InputProducerStage.

        Parameters:
            tiles_origin: Memory origin tag for the producer tile
                pipeline SMEM buffers.

        Args:
            a_tma_op: TMA descriptor used to multicast-load A tiles
                from GMEM.
            b_tma_op: TMA descriptor used to multicast-load B tiles
                from GMEM.
            sfa_tma_op: TMA descriptor used to copy SFA scale-factor
                tiles from GMEM.
            sfb_tma_op: TMA descriptor used to copy SFB scale-factor
                tiles from GMEM.
            tiles: Producer pipeline stage providing the SMEM tile
                slots to fill.
            peer_cta_coord: `(rank_n, rank_m, m_rank)` coordinates of
                the peer CTA within the cluster, used to compute
                per-CTA GMEM source offsets.
            work_tile_coord: `(m, n, batch)` tile coordinates within
                the current group, used to compute GMEM source
                offsets.
            a_multicast_mask: Bitmask of cluster CTAs receiving the
                A load.
            b_multicast_mask: Bitmask of cluster CTAs receiving the
                B load.
            iter_idx: K-tile iteration index within the current
                group's K loop.
            elect_one_cta: Whether this CTA is the elected CTA for
                the tile.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        var a_gmem_m_coord = (
            peer_m_rank * Self.a_tma_rows + work_tile_coord[0] * Self.BM
        )
        var b_gmem_n_coord = (
            peer_rank_m * Self.b_tma_rows
            + peer_rank_n * Self.BN
            + work_tile_coord[1] * Self.MMA_N
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                # Peer CTA slicing using TileTensor pattern (ptr + layout)
                var a_peer_tile = type_of(a_tile)(
                    a_tile._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tile.layout,
                )

                var k_coord = Int(iter_idx + j) * Self.BK

                # TileTensor directly to TMA (uses TileTensor overload)
                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, a_gmem_m_coord, batch_coord),
                    a_multicast_mask,
                )
                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, b_gmem_n_coord, batch_coord),
                    b_multicast_mask,
                )

                # TMA 5D now has TileTensor overload - pass tiles directly
                sfa_tma_op.async_copy_5d[Self.cta_group](
                    sfa_tile,
                    barrier[0],
                    (
                        0,
                        0,
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
                        work_tile_coord[0] * (Self.BM // SF_MN_GROUP_SIZE),
                        batch_coord,
                    ),
                )
                sfb_tma_op.async_copy_5d[Self.cta_group](
                    sfb_tile,
                    barrier[0],
                    (
                        0,
                        0,
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
                        work_tile_coord[1] * (Self.MMA_N // SF_MN_GROUP_SIZE),
                        batch_coord,
                    ),
                )

    # ========== MMA Operation ==========

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tiles: ConsumerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: Self.MmaOp,
        tmem_addr: UInt32,
        tmem_region: Self.TmemRegion,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations using ConsumerTiles.

        Parameters:
            tiles_origin: Memory origin tag for the consumer tile pipeline
                SMEM buffers (inferred).

        Args:
            tiles: Consumer pipeline stage providing the SMEM tile slots to
                consume.
            mma_op: Block-scaled MMA operation object that executes the
                tensor-core multiply-accumulate.
            tmem_addr: TMEM column address of the accumulator buffer for
                this output stage.
            tmem_region: TMEM region holding per-tile scale-factor offsets
                for SFA and SFB.
            iter_idx: K-tile iteration index within the current group's K
                loop.
            k_start: K-tile index where accumulation begins for the current
                group; the first iteration initializes the accumulator.
        """
        if elect_one_sync():
            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )

                var sfa_tmem_offset = UInt32(tmem_region.sfa(tile_idx).col_addr)
                var sfb_tmem_offset = UInt32(tmem_region.sfb(tile_idx).col_addr)

                var is_first_k = (iter_idx + j) == k_start

                # MMA has TileTensor overload - pass tiles directly
                # (layout is extracted from TileTensor type parameters)
                mma_op.mma(
                    a_tile,
                    b_tile,
                    sfa_tile,
                    sfb_tile,
                    tmem_addr,
                    sfa_tmem_offset,
                    sfb_tmem_offset,
                    init_c=is_first_k,
                )

            mma_op.commit(tiles.mbar())

    # ========== Epilogue ==========

    @staticmethod
    @always_inline
    def epilogue(
        c_tiles: Self.SmemType.Core.CTileArray,
        c_tma_op: Self.CTmaOp,
        stage: Self.TileWriterType.Stage,
        work_tile_coord: Tuple[UInt32, UInt32, UInt32],
        M: UInt32,
        N: UInt32,
        alpha: Float32 = Float32(1.0),
    ):
        """Execute epilogue to store accumulated results.

        Args:
            c_tiles: SMEM tile array holding C output tiles.
            c_tma_op: TMA descriptor for C store operations.
            stage: Output pipeline stage with accumulated results to store.
            work_tile_coord: `(m, n, batch)` tile coordinates within the
                current group.
            M: Row extent of the current group's output matrix.
            N: Column extent of the current group's output matrix.
            alpha: Scaling factor applied to accumulators before storing
                (defaults to 1.0).
        """
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))
        tile_writer.write_batched(
            c_tiles,
            stage,
            work_tile_coord,
            (M, N),
            alpha,
        )

    # ========== 2SM Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
    @__llvm_arg_metadata(a_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_template, `nvvm.grid_constant`)
    @__name(
        StaticString(
            Self.config.get_kernel_name()
            + "_fused_compute_epi" if Self.elementwise_compute_lambda_fn
            is not None else ""
        ),
    )
    def run_2sm(
        # Template tensormaps for SMEM initialization
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        # Per-block updatable tensormaps
        device_tma_a: Self.TMATensorTileArrayA,
        device_tma_b: Self.TMATensorTileArrayB,
        device_tma_sfa: Self.TMATensorTileArraySFA,
        device_tma_sfb: Self.TMATensorTileArraySFB,
        device_tma_c: Self.TMATensorTileArrayC,
        # Per-group pointer arrays (uint64 addresses)
        group_a_ptrs_lt: Self.GroupPtrTile,
        group_b_ptrs_lt: Self.GroupPtrTile,
        group_c_ptrs_lt: Self.GroupPtrTile,
        group_sfa_ptrs_lt: Self.GroupPtrTile,
        group_sfb_ptrs_lt: Self.GroupPtrTile,
        # Per-group problem sizes: (_num_groups, 4) with [M, N, K, L]
        problem_sizes_lt: Self.ProblemSizesTile,
        # Number of active groups
        num_groups: Int32,
    ):
        """Grouped block-scaled GEMM kernel with 2SM (cta_group=2) support.

        This entry point uses CLC-based work distribution for proper 2SM
        synchronization between CTAs in a cluster. Both CTAs cooperate on
        each tile, with one CTA doing MMA work and both doing TMA loads.

        Architecture matches the working block_scaled_matmul_kernel:
        - Scheduler warp: Produces work items via CLC barriers
        - TMA warp: Loads tiles with tensormap updates on group change
        - MMA warp: Waits on CLC, executes MMA (elected CTA only)
        - Epilogue warps: Stores results with tensormap updates

        Args:
            a_tma_template: Grid-constant TMA template for A loads, used
                to initialize the per-block A tensormap in SMEM.
            b_tma_template: Grid-constant TMA template for B loads, used
                to initialize the per-block B tensormap in SMEM.
            c_tma_template: Grid-constant TMA template for C stores, used
                to initialize the per-block C tensormap in SMEM.
            sfa_tma_template: Grid-constant TMA template for SFA loads,
                used to initialize the per-block SFA tensormap in SMEM.
            sfb_tma_template: Grid-constant TMA template for SFB loads,
                used to initialize the per-block SFB tensormap in SMEM.
            device_tma_a: Per-block updatable TMA descriptor array for A,
                indexed by `block_idx.x`.
            device_tma_b: Per-block updatable TMA descriptor array for B,
                indexed by `block_idx.x`.
            device_tma_sfa: Per-block updatable TMA descriptor array for
                SFA, indexed by `block_idx.x`.
            device_tma_sfb: Per-block updatable TMA descriptor array for
                SFB, indexed by `block_idx.x`.
            device_tma_c: Per-block updatable TMA descriptor array for C,
                indexed by `block_idx.x`.
            group_a_ptrs_lt: Per-group array of GMEM base addresses for
                the A matrices.
            group_b_ptrs_lt: Per-group array of GMEM base addresses for
                the B matrices.
            group_c_ptrs_lt: Per-group array of GMEM base addresses for
                the C matrices.
            group_sfa_ptrs_lt: Per-group array of GMEM base addresses for
                the SFA scale-factor tensors.
            group_sfb_ptrs_lt: Per-group array of GMEM base addresses for
                the SFB scale-factor tensors.
            problem_sizes_lt: Per-group problem sizes as a
                `(max_groups, 4)` tensor with `[M, N, K, L]` per group.
            num_groups: Number of active GEMM groups to process.
        """
        var _num_groups = Int(num_groups)
        Self.validate_config()

        # Alias kernel args for internal methods
        var group_a_ptrs = group_a_ptrs_lt
        var group_b_ptrs = group_b_ptrs_lt
        var group_c_ptrs = group_c_ptrs_lt
        var group_sfa_ptrs = group_sfa_ptrs_lt
        var group_sfb_ptrs = group_sfb_ptrs_lt
        var problem_sizes = problem_sizes_lt

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM accessors
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var clc_full = smem.pipelines.clc_full()
        var clc_empty = smem.pipelines.clc_empty()
        var clc_throttle = smem.pipelines.clc_throttle()
        var clc_response = smem.pipelines.clc_response()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Kernel Context =====
        var ctx = Self.Context(tmem_addr_storage)

        # ===== Initial Work Info =====
        # Compute initial work from first cluster's tile
        var initial_linear_idx = UInt32(
            ufloordiv(block_idx.x, 2)
        )  # 2SM: cta_group=2
        var initial_work = Self._compute_initial_work(
            problem_sizes, _num_groups, initial_linear_idx
        )

        # ===== Barrier Initialization =====
        Self.init_barriers_2sm(
            ctx,
            a_tma_template,
            b_tma_template,
            c_tma_template,
            sfa_tma_template,
            sfb_tma_template,
            input_barriers,
            accum_barriers,
            clc_throttle,
            clc_full,
            clc_empty,
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            var blk = block_idx.x
            var tensormap_init_done = False

            # Tensormap manager for SMEM descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )

            # Create CLC work iterator for TMA warp
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                _num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            with input_pipeline.producer() as producer:
                for current in work_iter:
                    # Wait for MMA warp to init tensormaps (first time only)
                    if not tensormap_init_done:
                        Self.TensormapAbInitBarrier.sync()
                        tensormap_init_done = True

                    # === LOOKAHEAD PATTERN (CuteDSL style) ===
                    # Initialize to "ready" (True), only peek if there's work.
                    var num_k_iters = Int(current.k_tile_count)
                    var next_ready = True
                    if num_k_iters > 0:
                        next_ready = producer.try_acquire()

                    # Update tensormaps on group change (overlaps with peek)
                    if current.group_changed:
                        tensormap_mgr.update_ab_for_group(
                            current.group_idx,
                            group_a_ptrs,
                            group_b_ptrs,
                            group_sfa_ptrs,
                            group_sfb_ptrs,
                            device_tma_a[blk],
                            device_tma_b[blk],
                            device_tma_sfa[blk],
                            device_tma_sfb[blk],
                        )

                    # Load tiles using lookahead pattern
                    for k_tile in range(num_k_iters):
                        with producer.acquire_if_needed(next_ready) as tiles:
                            Self.load_input_tiles(
                                device_tma_a[blk][],
                                device_tma_b[blk][],
                                device_tma_sfa[blk][],
                                device_tma_sfb[blk][],
                                tiles,
                                ctx.peer_cta_coord,
                                (
                                    Int(current.m),
                                    Int(current.n),
                                    0,
                                ),
                                ctx.a_multicast_mask,
                                ctx.b_multicast_mask,
                                UInt32(k_tile),
                                ctx.elect_one_cta,
                            )
                        # Peek for next iteration (CuteDSL style):
                        # Reset to ready, then conditionally peek.
                        next_ready = True
                        if k_tile + 1 < num_k_iters:
                            next_ready = producer.try_acquire()
                    syncwarp()

                producer.drain()

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # Create scheduler iterator for CLC work production
            var sched_iter = GroupedCLCSchedulerIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                _num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            for _ in sched_iter:
                sched_iter.signal_and_advance()

            sched_iter.drain()

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            # Initialize SMEM tensormaps from templates (MMA warp, per CuTe DSL)
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )
            tensormap_mgr.init_ab_tensormaps(
                a_tma_template,
                b_tma_template,
                sfa_tma_template,
                sfb_tma_template,
            )

            # Barrier sync with TMA warp
            Self.TensormapAbInitBarrier.sync()

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            # Create CLC work iterator for MMA warp
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                _num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
                use_clc_fetch=True,
            )

            with mma_ctx:
                for current in work_iter:
                    if ctx.elect_one_cta:
                        with mma_ctx.output_pipeline.producer() as output_stage:
                            var tmem_offset = UInt32(output_stage.tmem.offset())

                            with input_pipeline.consumer() as consumer:
                                var num_k_iters = Int(current.k_tile_count)

                                # === LOOKAHEAD PATTERN (CuteDSL style) ===
                                # Initialize to "ready" (True), only peek if there's work.
                                var next_ready = True
                                if num_k_iters > 0:
                                    next_ready = consumer.try_acquire()

                                for k_tile in range(num_k_iters):
                                    with consumer.acquire_if_needed(
                                        next_ready
                                    ) as input_tiles:
                                        Self.mma(
                                            input_tiles,
                                            mma_op,
                                            tmem_offset,
                                            tmem_region,
                                            UInt32(k_tile),
                                            0,
                                        )
                                    # Peek for next iteration (CuteDSL style):
                                    # Reset to ready, then conditionally peek.
                                    next_ready = True
                                    if k_tile + 1 < num_k_iters:
                                        next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            var blk = block_idx.x

            # Tensormap manager for C descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_b).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfa).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_sfb).as_unsafe_any_origin(),
                    UnsafePointer(to=smem.tensormap_c).as_unsafe_any_origin(),
                ),
            )

            # Init C tensormap in SMEM
            tensormap_mgr.init_c_tensormap(c_tma_template)
            syncwarp()

            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            # Create work iterator for epilogue (uses simple advance, not CLC)
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                _num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            with epi_ctx:
                for current in work_iter:
                    # Update C tensormap on group change
                    if current.group_changed:
                        tensormap_mgr.update_c_for_group(
                            current.group_idx,
                            group_c_ptrs,
                            device_tma_c[blk],
                        )

                    # Get current group's M, N dimensions
                    var g = Int(current.group_idx)
                    var group_m = UInt32(Int(problem_sizes[g, 0]))
                    var group_n = UInt32(Int(problem_sizes[g, 1]))

                    with epi_ctx.output_pipeline.consumer() as output_stage:
                        Self.epilogue(
                            c_tiles,
                            device_tma_c[blk][],
                            output_stage,
                            (current.m, current.n, UInt32(0)),
                            group_m,
                            group_n,
                        )

    @staticmethod
    @always_inline
    def _compute_initial_work(
        problem_sizes: Self.ProblemSizesTile,
        _num_groups: Int,
        linear_idx: UInt32,
    ) -> GroupedWorkInfo:
        """Compute initial work info from linear tile index."""
        # Build cumulative tiles
        var cumulative = StaticTuple[UInt32, Self.max_groups + 1]()

        comptime for i in range(Self.max_groups + 1):
            cumulative[i] = 0

        var cumsum: UInt32 = 0
        for g in range(_num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var m_tiles = ceildiv(Int(m), Self.BM)
            var n_tiles = ceildiv(Int(n), Self.BN)
            cumsum += UInt32(m_tiles * n_tiles)
            cumulative[g + 1] = cumsum

        if linear_idx >= cumsum:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = UInt32(_num_groups)
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < cumulative[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        var local_idx = linear_idx - cumulative[Int(group_idx)]
        var m = UInt32(Int(problem_sizes[Int(group_idx), 0]))
        var k = UInt32(Int(problem_sizes[Int(group_idx), 2]))
        var m_tiles = ceildiv(Int(m), Self.BM)
        var k_tiles = ceildiv(Int(k), Self.BK)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=True,  # First tile always triggers update
        )
