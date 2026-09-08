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
"""Provides SM100 (Blackwell) MMA operation structs for warp-specialized GEMM kernels using UMMA instructions."""

from std.sys import size_of
from std.math import align_up

from max.gpu.primitives.cluster import cluster_mask_base
from max.gpu import block_id_in_cluster
from max.gpu.host._tensormap import SwizzleMode
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.arch.mma_nvidia_sm100 import *
from max.gpu.compute.arch.tcgen05 import *
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout import IntTuple, Layout, TileTensor
from layout.tile_layout import TensorLayout, _types_to_int_tuple
from layout.tensor_core_async import (
    _CM_ROW_BYTES,
    tile_to_descriptor,
    tile_layout_k_major,
    tile_layout_mn_major,
)

from std.utils.index import Index, IndexList, product
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    block_scaled_operands_compatible,
)


def _create_mma_desc_k_major[
    dtype: DType, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[Scalar[dtype], address_space=.SHARED, ...]
) -> MMASmemDescriptor:
    """Creates an MMA descriptor for K-major layout directly from swizzle mode.

    Bypasses the legacy Layout pipeline by computing SBO/LBO from swizzle
    parameters. For K-major: SBO = 8 * swizzle_mode.bytes(),
    LBO = 16 bytes (core matrix row size).
    """
    comptime SBO = 8 * swizzle_mode.bytes()
    comptime LBO = _CM_ROW_BYTES
    return MMASmemDescriptor.create[SBO, LBO, swizzle_mode](ptr)


@fieldwise_init("implicit")
struct Major(TrivialRegisterPassable):
    """Selects the major (contiguous) dimension for an MMA operand tile layout.

    Used to configure whether the K dimension or the MN dimension is the
    innermost (contiguous) axis of the tile in shared memory.
    """

    var val: Int

    comptime K = Major(0)
    comptime MN = Major(1)

    def __eq__(self, rhs: Major) -> Bool:
        return self.val == rhs.val


def max_contiguous_tile_shape[
    rank: Int,
    //,
    dtype: DType,
    tile_shape: IndexList[rank],
    /,
    *,
    major: Major = Major.K,
    swizzle_mode: SwizzleMode = SwizzleMode.NONE,
]() -> IntTuple:
    """Returns the maximum shape of a tile that's contiguous in memory for mma op. This is used to create TMA descriptor.
    """

    comptime assert rank == 2, "Only 2D tensors are supported!"

    comptime if major == Major.K:
        # Tile shape is (MN, K), max K is based on swizzle.
        return IntTuple(tile_shape[0], swizzle_mode.bytes() // size_of[dtype]())
    elif major == Major.MN:
        # Tile shape is (K, MN), max MN is based on swizzle, max K is 8 based on
        # canonical layout.
        # The following are rare in practice but worth checking.
        # TODO: this may not work for swizzle.NONE, need to double-check
        # TODO: for MN = swizzle_bytes // sizeof,  tile_shape[0] may be the max
        return IntTuple(8, swizzle_mode.bytes() // size_of[dtype]())
    else:
        comptime assert False, "Invalid major"


def _create_mma_desc_pair[
    dtype: DType, //, canonical_layout: Layout, swizzle_mode: TensorMapSwizzle
](
    ptr: UnsafePointer[Scalar[dtype], address_space=.SHARED, ...]
) -> MMASmemDescriptorPair:
    # Extract the stride values from the canonical layout
    # The canonical layout is expected to have at least 2 dimensions
    comptime stride01 = canonical_layout[0].stride[1].value()
    comptime stride11 = canonical_layout[1].stride[1].value()
    comptime SBO = stride01 * size_of[dtype]()
    comptime LBO = stride11 * size_of[dtype]()

    # Create and return the MMA shared memory descriptor
    # This will be used by the SM100 MMA operations to access shared memory
    return MMASmemDescriptorPair.create[SBO, LBO, swizzle_mode](ptr)


@always_inline
def smem_descriptor[
    dtype: DType,
    //,
    *,
    BMN: Int,
    BK: Int,
    swizzle_mode: TensorMapSwizzle,
    is_k_major: Bool,
    page_dense: Bool = False,
](
    ptr: UnsafePointer[Scalar[dtype], address_space=.SHARED, ...]
) -> MMASmemDescriptorPair:
    """Creates an MMASmemDescriptorPair for an SM100 MMA operand tile in shared memory.

    Selects either K-major or MN-major tile layout based on `is_k_major`, builds the
    canonical descriptor layout, and creates the corresponding pair of SMEM descriptors
    used by the SM100 UMMA instructions.

    Parameters:
        dtype: Element type of the operand.
        BMN: Block tile size along the M or N dimension.
        BK: Block tile size along the K dimension.
        swizzle_mode: Swizzle mode for the shared memory access pattern.
        is_k_major: When True, uses K-major (A/Q@K') layout; when False, uses
            MN-major (V/P@V) layout.
        page_dense: When True, selects the page-dense (row-major atoms) variant
            of the tile layout for the SM100 row-major page-fold path.

    Args:
        ptr: Pointer into shared memory for the operand tile.

    Returns:
        An `MMASmemDescriptorPair` ready for use with SM100 UMMA instructions.
    """
    # `page_dense` selects the native chunk-inner (row-major atoms) layout
    # (SM100 row-major page-fold path) for the corresponding operand frame:
    # k-major (`is_k_major=True`, K / Q@K') and mn-major (`is_k_major=False`,
    # V / P@V) each have their own page-dense branch in their `tile_layout_*`.
    comptime smem_layout = tile_layout_k_major[
        dtype, BMN, BK, swizzle_mode, page_dense=page_dense
    ]() if is_k_major else tile_layout_mn_major[
        dtype, BMN, BK, swizzle_mode, page_dense=page_dense
    ]()
    comptime canonical_layout = tile_to_descriptor[
        dtype, smem_layout, is_k_major=is_k_major
    ]()
    comptime cl = canonical_layout if is_k_major else canonical_layout.transpose()
    return _create_mma_desc_pair[
        canonical_layout=cl, swizzle_mode=swizzle_mode
    ](ptr)


struct MmaOpSM100_SS[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    /,
    *,
    accum_type: DType = .float32,
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = False,
](Defaultable, TrivialRegisterPassable):
    """SM100 (Blackwell) warp-specialized MMA operation for standard (non-block-scaled) GEMM.

    Encapsulates a UMMA instruction descriptor and multicast mask for issuing
    asynchronous tensor-core MMA operations from shared memory to tensor memory
    (TMEM) on NVIDIA SM100 GPUs. Supports optional CTA clustering for multicast.

    Parameters:
        c_type: Output accumulator element type.
        a_type: A operand element type.
        b_type: B operand element type; must match `a_type`.
        block_tile_shape: (BM, BN, BK) shape of the shared-memory tile processed
            per CTA.
        mma_shape: (MMA_M, MMA_N, MMA_K) shape of a single UMMA instruction.
        accum_type: Accumulator precision; defaults to float32.
        cta_group: Number of CTAs collaborating on one tile (1 or 2).
        cluster_shape: CTA cluster shape for multicast; defaults to (1, 1, 1).
        a_swizzle: Swizzle mode for the A operand shared-memory layout.
        b_swizzle: Swizzle mode for the B operand shared-memory layout.
        transpose_b: Must be True; SM100 UMMA always uses transposed B.
    """

    var idesc: UMMAInsDescriptor[Self._get_umma_kind[Self.a_type]()]
    var mask: UInt16

    @always_inline
    def __init__(out self):
        comptime assert (
            Self.transpose_b
        ), "MmaOpSM100 only supports transposed B"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "MmaOpSM100 only supports cta_group 1 or 2"
        comptime assert (
            Self.a_type == Self.b_type
        ), "a_type and b_type must be the same"
        comptime assert (
            Self.mma_shape[2] == 32 // size_of[Self.a_type]()
        ), "MMA_K must be 32 // size_of(a_type) (16 for 16-bit, 32 for 8-bit)"

        self.idesc = UMMAInsDescriptor[
            Self._get_umma_kind[Self.a_type]()
        ].create[
            Self.accum_type,
            Self.a_type,
            Self.b_type,
            Index[dtype=DType.uint32](Self.mma_shape[0], Self.mma_shape[1]),
            transpose_b=Self.transpose_b,
        ]()

        self.mask = 0

        # Here we compute the mask inside mma object to hide the complexity.
        # We may get better asm if the mask if computed outside from TMA masks,
        # and passed to `commit`, need to verify.
        comptime if product(Self.cluster_shape) > 1:
            comptime dim0_mask = cluster_mask_base[Self.cluster_shape, 0]()
            comptime dim1_mask = cluster_mask_base[Self.cluster_shape, 1]()

            # The mask includes ctas on the same row and column in the cluster
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             o x o o
            #             o x o o
            #             o x o o
            self.mask = (
                dim0_mask
                << UInt16(block_id_in_cluster.y * Self.cluster_shape[0])
            ) | (dim1_mask << UInt16(block_id_in_cluster.x))

            # Include peer cta's row
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             x x x x
            #             o x o o
            #             o x o o
            comptime if Self.cta_group == 2:
                self.mask |= dim1_mask << UInt16(block_id_in_cluster.x ^ 1)

    @always_inline
    def make_a_desc(
        self,
        a: TileTensor[address_space=.SHARED, ...],
    ) -> MMASmemDescriptor:
        """Build the K-major MMA descriptor for an A operand tile.

        Exposed so callers that issue MMAs over several adjacent SMEM tiles
        (e.g. a k-group loop) can build the descriptor once and advance it by
        a comptime byte stride per tile, rather than rebuilding the full
        descriptor (base-pointer mask + bitfield inserts) for each tile.

        Args:
            a: A operand tile in shared memory.

        Returns:
            The K-major MMA shared-memory descriptor for `a`.
        """
        return _create_mma_desc_k_major[a.dtype, Self.a_swizzle](a.ptr)

    @always_inline
    def make_b_desc(
        self,
        b: TileTensor[address_space=.SHARED, ...],
    ) -> MMASmemDescriptor:
        """Build the K-major MMA descriptor for a B operand tile.

        See `make_a_desc` for why this is exposed.

        Args:
            b: B operand tile in shared memory.

        Returns:
            The K-major MMA shared-memory descriptor for `b`.
        """
        return _create_mma_desc_k_major[b.dtype, Self.b_swizzle](b.ptr)

    @always_inline
    def mma(
        self,
        a: TileTensor[address_space=.SHARED, ...],
        b: TileTensor[address_space=.SHARED, ...],
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """Issue MMA operations over K tiles from shared memory to TMEM.

        Args:
            a: A operand tile in shared memory.
            b: B operand tile in shared memory.
            c_tmem: TMEM address for the accumulator.
            init_c: When True, zero-initialize the accumulator on the first
                K slice instead of accumulating.
        """
        self.mma_from_desc(
            self.make_a_desc(a), self.make_b_desc(b), c_tmem, init_c
        )

    @always_inline
    def mma_from_desc(
        self,
        a_desc: MMASmemDescriptor,
        b_desc: MMASmemDescriptor,
        c_tmem: UInt32,
        init_c: Bool,
    ):
        """Issue MMA operations over K tiles from precomputed SMEM descriptors.

        Identical to `mma`, but takes already-built A/B descriptors. This lets
        a k-group caller build the descriptor base once and advance it by a
        comptime byte stride per tile (the SM100 SMEM-descriptor base address
        adds linearly), avoiding a redundant full descriptor rebuild per tile.

        Args:
            a_desc: Precomputed K-major descriptor for the A operand tile.
            b_desc: Precomputed K-major descriptor for the B operand tile.
            c_tmem: TMEM address for the accumulator.
            init_c: When True, zero-initialize the accumulator on the first
                K slice instead of accumulating.
        """
        # K-major swizzle layout: within a swizzle tile (k < sw_K), elements
        # are stride-1. Across swizzle tile boundaries (k >= sw_K), the
        # offset jumps by rows * sw_K elements to the next swizzle group.
        comptime sw_K_a = Self.a_swizzle.bytes() // size_of[Self.a_type]()
        comptime sw_K_b = Self.b_swizzle.bytes() // size_of[Self.b_type]()
        comptime BM = Self.block_tile_shape[0]
        comptime BN = Self.block_tile_shape[1]

        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = (
                (k % sw_K_a) + (k // sw_K_a) * BM * sw_K_a
            ) * size_of[Self.a_type]()
            comptime b_offset = (
                (k % sw_K_b) + (k // sw_K_b) * BN * sw_K_b
            ) * size_of[Self.b_type]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            mma[Self.cta_group](
                a_desc + a_offset,
                b_desc + b_offset,
                c_tmem,
                self.idesc,
                c_scale=c_scale,
            )

    @always_inline
    def commit(
        self,
        ptr_mbar: UnsafePointer[address_space=.SHARED, ...],
    ):
        comptime if product(Self.cluster_shape) == 1:
            mma_arrive[Self.cta_group](ptr_mbar)
        else:
            mma_arrive_multicast[Self.cta_group](ptr_mbar, self.mask)

    @always_inline
    def wait(self):
        pass

    @staticmethod
    def _get_umma_kind[dtype: DType]() -> UMMAKind:
        comptime if dtype == .float32:
            return UMMAKind.KIND_TF32
        elif dtype in (DType.float16, DType.bfloat16):
            return UMMAKind.KIND_F16
        elif dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
            return UMMAKind.KIND_F8F6F4
        else:
            comptime assert False, String(
                "Unsupported/not implemented operand type for UMMA: ",
                String(dtype),
            )

        return UMMAKind(-1)


struct MmaOpSM100_BlockScaled_SS[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    scaling_kind: UMMAKind,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    /,
    *,
    accum_type: DType = .float32,
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = False,
    enable_small_sfb: Bool = False,
](Defaultable, TrivialRegisterPassable):
    """SM100 (Blackwell) warp-specialized MMA operation for block-scaled GEMM (MXFP4, MXFP8, NVFP4).

    Extends `MmaOpSM100_SS` with per-block scale factor handling for quantized
    formats. Copies scale factors from shared memory to TMEM via `tcgen05_cp`
    and issues UMMA instructions with the appropriate scale index per K slice.

    Parameters:
        c_type: Output accumulator element type.
        a_type: A operand element type (float8_e4m3fn or uint8 for FP4).
        b_type: B operand element type; must match `a_type`.
        sfa_dtype: Scale factor dtype for A; must match `sfb_dtype`.
        sfb_dtype: Scale factor dtype for B.
        scaling_kind: UMMA scaling kind (MXF8F6F4, MXF4, or MXF4NVF4).
        block_tile_shape: (BM, BN, BK) shape of the shared-memory tile.
        mma_shape: (MMA_M, MMA_N, MMA_K) shape of a single UMMA instruction.
        accum_type: Accumulator precision; defaults to float32.
        cta_group: Number of CTAs collaborating on one tile (1 or 2).
        cluster_shape: CTA cluster shape for multicast; defaults to (1, 1, 1).
        a_swizzle: Swizzle mode for the A operand.
        b_swizzle: Swizzle mode for the B operand.
        transpose_b: Must be True; SM100 UMMA always uses transposed B.
        enable_small_sfb: Allow SFB loading for MMA_N < 64 via cooperative tcgen05_st.
    """

    var idesc: UMMAInsDescriptor[Self.scaling_kind]
    var mask: UInt16

    @always_inline
    def __init__(out self):
        comptime assert Self.scaling_kind in (
            UMMAKind.KIND_MXF8F6F4,
            UMMAKind.KIND_MXF4,
            UMMAKind.KIND_MXF4NVF4,
        ), (
            "Only support MXF8F6F4, MXF4, or MXF4NVF4 scaling kind for"
            " block scaled matmul!"
        )
        comptime assert (
            Self.transpose_b
        ), "MmaOpSM100 only supports transposed B"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "MmaOpSM100 only supports cta_group 1 or 2"
        comptime assert block_scaled_operands_compatible[
            Self.a_type, Self.b_type
        ](), "a_type and b_type must be the same, or the W4A8 pair"
        comptime assert Self.a_type in (
            DType.float8_e4m3fn,
            DType.uint8,  # TODO: (KERN-2238) replace with FP4-E2M1
        ), "Only support float8_e4m3fn or uint8 (F4-E2M1) for input operands"
        comptime assert (
            Self.sfa_dtype == Self.sfb_dtype
        ), "sfa_dtype and sfb_dtype must be the same"
        comptime assert Self.sfa_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e8m0fnu,
        ), "Only support float8_e4m3fn or float8_e8m0fnu for scales"

        self.idesc = UMMAInsDescriptor[Self.scaling_kind].create[
            Self.accum_type,
            Self.a_type,
            Self.b_type,
            Self.sfa_dtype,
            Index[dtype=DType.uint32](Self.mma_shape[0], Self.mma_shape[1]),
            transpose_b=Self.transpose_b,
        ]()

        self.mask = 0

        # Here we compute the mask inside mma object to hide the complexity.
        # We may get better asm if the mask if computed outside from TMA masks,
        # and passed to `commit`, need to verify.
        comptime if product(Self.cluster_shape) > 1:
            comptime dim0_mask = cluster_mask_base[Self.cluster_shape, 0]()
            comptime dim1_mask = cluster_mask_base[Self.cluster_shape, 1]()

            # The mask includes ctas on the same row and column in the cluster
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             o x o o
            #             o x o o
            #             o x o o
            self.mask = (
                dim0_mask
                << UInt16(block_id_in_cluster.y * Self.cluster_shape[0])
            ) | (dim1_mask << UInt16(block_id_in_cluster.x))

            # Include peer cta's row
            # Example mask for cta (0, 1) is cluster (4,4)
            #             x x x x
            #             x x x x
            #             o x o o
            #             o x o o
            comptime if Self.cta_group == 2:
                self.mask |= dim1_mask << UInt16(block_id_in_cluster.x ^ 1)

    @always_inline
    def mma(
        self,
        a: TileTensor[address_space=.SHARED, ...],
        b: TileTensor[address_space=.SHARED, ...],
        sfa_smem: TileTensor[address_space=.SHARED, ...],
        sfb_smem: TileTensor[address_space=.SHARED, ...],
        c_tmem: UInt32,
        sfa_tmem: UInt32,
        sfb_tmem: UInt32,
        init_c: Bool,
        sfb_tmem_adj: UInt32 = UInt32(
            0
        ),  # TMEM col offset for MMA_N < SF_MN_GROUP_SIZE
    ):
        """TileTensor overload for block-scaled MMA input tiles.

        Creates MMA descriptors directly from swizzle parameters.
        """

        # A/B descriptors: compute directly from swizzle mode (no legacy Layout).
        var a_desc = _create_mma_desc_k_major[a.dtype, Self.a_swizzle](a.ptr)
        var b_desc = _create_mma_desc_k_major[b.dtype, Self.b_swizzle](b.ptr)

        comptime assert (
            Self.block_tile_shape[2] == 128 and Self.mma_shape[2] == 32
        ), "block_tile_shape[2] must be 128 and mma_shape[2] must be 32"

        # For MMA_N < 64, tcgen05_cp cannot be used for SFB because:
        # 1. It always writes 4 TMEM columns (one full SF_MN_GROUP)
        # 2. UMMA reads SFB from dp 0..MMA_N-1 of the given column
        # 3. Odd column addresses cause MISALIGNED_ADDRESS crashes
        # The caller must pre-load SFB into TMEM via tcgen05_st from
        # warps covering dp 0-127.  sfb_tmem_adj is ignored (always 0).

        comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
            # when scaling kind is MXF8F6F4, one scale tile covers the whole [BM,BK] and [MMA_N,BK] tiles so we load it once.
            self._copy_sf_to_tmem_tt[
                Self.sfa_dtype, sfa_smem.LayoutType, Self.block_tile_shape[0], 0
            ](sfa_smem, sfa_tmem)
            comptime if Self.mma_shape[1] % 64 == 0 or Self.enable_small_sfb:
                self._copy_sf_to_tmem_tt[
                    Self.sfb_dtype,
                    sfb_smem.LayoutType,
                    align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE),
                    0,
                ](sfb_smem, sfb_tmem)
        elif Self.scaling_kind == UMMAKind.KIND_MXF4:
            # MXF4: each SF tile covers 2 K-slices, so 2 tiles for 4 slices.
            comptime num_mxf4_sf_tiles = Self.block_tile_shape[2] // (
                2 * Self.mma_shape[2]
            )
            comptime assert (
                num_mxf4_sf_tiles == 2
            ), "MXF4 expects 2 SF tiles for BK=128"
            comptime for sf_k_tile in range(num_mxf4_sf_tiles):
                self._copy_sf_to_tmem_tt[
                    Self.sfa_dtype,
                    sfa_smem.LayoutType,
                    Self.block_tile_shape[0],
                    sf_k_tile,
                ](sfa_smem, sfa_tmem)
                # Only use tcgen05_cp for SFB when MMA_N meets TMEM
                # alignment (MMA_N % 64 == 0).  For smaller MMA_N the
                # caller loads SFB externally via tcgen05_st.
                comptime if Self.mma_shape[1] % 64 == 0:
                    # Offset sfb_tmem per K-tile to avoid TMEM address
                    # collision when MMA_N > SF_MN_GROUP_SIZE (2+ MN groups).
                    comptime sfb_k_offset = sf_k_tile * (
                        (
                            align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE)
                            - SF_MN_GROUP_SIZE
                        )
                        // 32
                    )
                    self._copy_sf_to_tmem_tt[
                        Self.sfb_dtype,
                        sfb_smem.LayoutType,
                        align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE),
                        sf_k_tile,
                    ](sfb_smem, sfb_tmem + UInt32(sfb_k_offset))

        # K-iteration: offset = k * sizeof (contiguous within swizzle tile).
        # Both operands span one byte per element here: the FP4 TMA copy pads
        # an E2M1 operand into shared memory (8 packed bytes then an 8-byte
        # gap per 16 values), so a K offset scales the same for either dtype.
        comptime for k in range(0, Self.block_tile_shape[2], Self.mma_shape[2]):
            comptime a_offset = k * size_of[Self.a_type]()
            comptime b_offset = k * size_of[Self.b_type]()

            var c_scale: UInt32 = UInt32(0) if (init_c and k == 0) else UInt32(
                1
            )

            comptime sf_idx = k // Self.mma_shape[2]

            comptime if Self.scaling_kind == UMMAKind.KIND_MXF8F6F4:
                var runtime_desc = UMMAInsDescriptor[
                    Self.scaling_kind
                ].update_desc_with_sf_id[UInt32(sf_idx)](
                    self.idesc,
                )
                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    runtime_desc,
                    sfa_tmem,
                    sfb_tmem + sfb_tmem_adj,
                    c_scale=c_scale,
                )
            elif Self.scaling_kind == UMMAKind.KIND_MXF4:
                # MXF4 maps K-slices (sf_idx: 0..3) to:
                # - SF tile index: 0,0,1,1
                # - SF id in descriptor: 0,2,0,2
                comptime sfa_tile_stride = SF_MN_GROUP_SIZE // 32
                comptime sfb_tile_stride = align_up(
                    Self.mma_shape[1], SF_MN_GROUP_SIZE
                ) // 32
                comptime sf_tile_idx = sf_idx // 2
                comptime sf_id_in_tile = (sf_idx % 2) * 2
                var runtime_desc = UMMAInsDescriptor[
                    Self.scaling_kind
                ].update_desc_with_sf_id[UInt32(sf_id_in_tile)](
                    self.idesc,
                )
                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    runtime_desc,
                    sfa_tmem + UInt32(sf_tile_idx * sfa_tile_stride),
                    sfb_tmem
                    + sfb_tmem_adj
                    + UInt32(sf_tile_idx * sfb_tile_stride),
                    c_scale=c_scale,
                )
            else:
                # when scaling kind is MXFP4NVF4, four scale tiles cover the whole [BM,BK] and [MMA_N,BK] tiles so we need to load one scale tile for each k iteration.
                self._copy_sf_to_tmem_tt[
                    Self.sfa_dtype,
                    sfa_smem.LayoutType,
                    Self.block_tile_shape[0],
                    sf_idx,
                ](sfa_smem, sfa_tmem)
                comptime if Self.mma_shape[
                    1
                ] % 64 == 0 or Self.enable_small_sfb:
                    self._copy_sf_to_tmem_tt[
                        Self.sfb_dtype,
                        sfb_smem.LayoutType,
                        align_up(Self.mma_shape[1], SF_MN_GROUP_SIZE),
                        sf_idx,
                    ](sfb_smem, sfb_tmem)

                mma[Self.cta_group](
                    a_desc + a_offset,
                    b_desc + b_offset,
                    c_tmem,
                    self.idesc,
                    sfa_tmem + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    sfb_tmem
                    + sfb_tmem_adj
                    + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    c_scale=c_scale,
                )

    @always_inline
    def commit(
        self,
        ptr_mbar: UnsafePointer[address_space=.SHARED, ...],
    ):
        comptime if product(Self.cluster_shape) == 1:
            mma_arrive[Self.cta_group](ptr_mbar)
        else:
            mma_arrive_multicast[Self.cta_group](ptr_mbar, self.mask)

    @always_inline
    def wait(self):
        pass

    @always_inline
    def _copy_sf_to_tmem_tt[
        sf_dtype: DType,
        SFLayoutType: TensorLayout,
        TILE_MN: Int,
        tile_k_idx: Int,
    ](self, sf_smem: TileTensor[address_space=.SHARED, ...], sf_tmem: UInt32,):
        """TileTensor overload for copying scale factors to TMEM via tcgen05_cp.

        Only valid for MMA_N % 64 == 0.  For smaller MMA_N, the caller
        must load SFB externally via cooperative tcgen05_st.

        Args:
            sf_smem: Scale factor tile in shared memory.
            sf_tmem: TMEM column address for scale factors.
        """
        comptime sf_smem_layout = Layout(
            _types_to_int_tuple[SFLayoutType._shape_types](),
            _types_to_int_tuple[SFLayoutType._stride_types](),
        )

        comptime for i in range(TILE_MN // SF_MN_GROUP_SIZE):
            comptime idx = IntTuple(
                i * SF_ATOM_M[0], tile_k_idx * SF_ATOM_M[1] * SF_ATOM_K
            )
            comptime sf_offset = sf_smem_layout(idx) * size_of[sf_dtype]()
            var sf_tmem_addr = (
                sf_tmem
                + UInt32(i * (SF_MN_GROUP_SIZE // 32))
                + UInt32(tile_k_idx * (SF_MN_GROUP_SIZE // 32))
            )
            var sf_desc = MMASmemDescriptor.create[
                8 * 16, 0, TensorMapSwizzle.SWIZZLE_NONE
            ](sf_smem.ptr + sf_offset)
            tcgen05_cp[
                cta_group=Int32(Self.cta_group),
                datapaths=32,
                bits=128,
                multicast="warpx4",
            ](sf_tmem_addr, sf_desc)
