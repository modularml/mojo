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
"""SM100 matmul configuration types and utilities.

This module provides configuration structs for SM100 (Blackwell) GPU matmul
operations, including standard matmul and block-scaled matmul variants.

Two config types share 17 common fields (tile shapes, pipeline stages,
swizzle modes, etc.) and the same `__init__` helpers (`_compute_block_tile_shape`,
`_compute_output_tile_shape`, `_compute_swizzle_modes`, `_maximize_pipeline_stages`).
BlockScaledMatmulConfig extends this with 3 scaling-specific fields
(`scaling_kind`, `vec_sf_size`, `num_sf_k_tiles`). Each has its own heuristic
(`choose_config` / `choose_block_scaled_config`) because the valid MMA shape
ranges and alignment requirements differ between standard and scaled kernels.
"""

from std.collections.set import Set
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.host.info import B200
from std.itertools.itertools import product
from layout.tensor_core import get_mma_shape
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.sys import size_of
from std.math import align_down, align_up, ceildiv
from ...tile_scheduler import RasterOrder
from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    block_scaled_operands_compatible,
    block_scaled_umma_kind,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


@fieldwise_init("implicit")
struct GEMMKind(Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Struct for GEMM types.

    This struct defines the different types of GEMM that is supported by BlackWell Such as BMM, GEMM, GMM, etc.
    """

    var _value: Int32

    comptime GEMM = Self(0)
    """GEMM type."""

    comptime BMM = Self(1)
    """BMM type."""

    comptime GMM = Self(2)
    """GMM type."""

    comptime BLOCK_SCALED_1D2D_FP8 = Self(3)
    """BLOCK_SCALED_1D2D_FP8 type."""

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """Convert GEMM kind to an integer value.

        Returns:
            The integer value representing the GEMM type.
        """
        return Int(self._value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Check if two GEMM kinds are equal.

        Args:
            other: The other GEMM kind to compare with.

        Returns:
            True if the GEMM kinds are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        """Check if two GEMM kinds are not equal.

        Args:
            other: The other GEMM kind to compare with.

        Returns:
            True if the GEMM kinds are not equal, False otherwise.
        """
        return self._value != other._value

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write the GEMM kind to a writer.

        Args:
            writer: The writer to write the GEMM kind to.
        """
        if self == Self.GEMM:
            writer.write("kind::gemm")
        elif self == Self.BMM:
            writer.write("kind::bmm")
        elif self == Self.GMM:
            writer.write("kind::gmm")
        elif self == Self.BLOCK_SCALED_1D2D_FP8:
            writer.write("kind::block_scaled_1d2d_fp8")
        else:
            writer.write("kind::unknown")

    @always_inline
    def __str__(self) -> String:
        """Convert GEMM kind to a string."""
        if self == Self.GEMM:
            return "gemm"
        elif self == Self.BMM:
            return "bmm"
        elif self == Self.GMM:
            return "gmm"
        elif self == Self.BLOCK_SCALED_1D2D_FP8:
            return "block_scaled_1d2d_fp8"
        else:
            return "unknown"


# ============================================================================
# Output Pipeline Configuration
# ============================================================================


@fieldwise_init
struct OutputPipelineConfig(Copyable, Equatable, TrivialRegisterPassable):
    """Configuration for the MMA-to-Epilogue output pipeline.

    Bundles the three parameters that jointly define TMEM accumulator
    stage management for MMA/epilogue synchronization:
    - num_stages: Number of accumulator pipeline stages (typically 1 or 2).
    - stage_stride_cols: TMEM column stride between accumulator stages.
    - cta_group: CTA group size (1 or 2).

    **stage_stride_cols computation**: Two strategies are used depending on
    the kernel family:
    - Standard kernels (default, blockwise_fp8): `NUM_TMEM_COLS // num_stages`
      (= 512 // stages). Divides all 512 TMEM columns evenly among stages.
    - Block-scaled kernels (block_scaled, grouped, 1d1d variants): `MMA_N`.
      Sizes each stage to match the MMA output width, which may be smaller
      than half of TMEM when MMA_N < 256.

    Constructed once per kernel struct and propagated to all pipeline
    types (OutputTilePipeline, warp contexts, TileWriter, etc.).
    """

    var num_stages: Int
    var stage_stride_cols: Int
    var cta_group: Int


# ============================================================================
# Shared Configuration Helpers
# ============================================================================


def _compute_block_tile_shape[
    a_type: DType
](mma_shape: IndexList[3], cta_group: Int) -> IndexList[3]:
    """Compute block tile shape from MMA shape and CTA group."""
    return Index(
        mma_shape[0] // cta_group,
        mma_shape[1] // cta_group,
        128 // size_of[a_type](),
    )


def _compute_output_tile_shape(
    c_type: DType, mma_shape: IndexList[3], cta_group: Int, AB_swapped: Bool
) -> IndexList[2]:
    """Compute output tile shape based on MMA config."""
    # If MMA_M is 256, each of the pair ctas has the entire MMA_N.
    # If MMA_M is 128, each of the pair ctas has 1/2 of MMA_N.
    # If cta_group=1, the cta has the entire MMA_N.

    # MMA_M=128/256 cta_group=2 all use 128 rows in output tile.
    var output_tile_m = 128 if cta_group == 2 else mma_shape[0]

    if c_type == .bfloat16:
        var c_tile_n = mma_shape[1] if (
            mma_shape[0] == 256 or cta_group == 1
        ) else (mma_shape[1] // 2)
        var output_tile_n = 8
        # For AB_swapped, c_swizzle is picked independently of output_tile_n
        # (see _compute_swizzle_modes). This in turn limits the chosen TMA
        # op (see c_tile_dim1 in matmul_kernels.mojo).
        if c_tile_n % 64 == 0 and not AB_swapped:
            output_tile_n = 64
        elif c_tile_n % 32 == 0:
            output_tile_n = 32
        elif c_tile_n % 16 == 0:
            output_tile_n = 16
        return Index(output_tile_n, output_tile_m) if AB_swapped else Index(
            output_tile_m, output_tile_n
        )
    elif c_type == .float32:
        var c_tile_n = mma_shape[1] if (
            mma_shape[0] == 256 or cta_group == 1
        ) else (mma_shape[1] // 2)
        var output_tile_n = 8
        if c_tile_n % 32 == 0 and not AB_swapped:
            output_tile_n = 32
        elif c_tile_n % 16 == 0:
            output_tile_n = 16
        return Index(output_tile_n, output_tile_m) if AB_swapped else Index(
            output_tile_m, output_tile_n
        )
    else:  # FP8 output tile shape
        var output_tile_n = 16  # no swizzle for fp8 output dtype
        return Index(output_tile_n, output_tile_m) if AB_swapped else Index(
            output_tile_m, output_tile_n
        )


def _compute_swizzle_modes(
    c_type: DType,
    output_tile_shape: IndexList[2],
    AB_swapped: Bool,
    is_gmm: Bool = False,
) -> Tuple[TensorMapSwizzle, TensorMapSwizzle, TensorMapSwizzle]:
    """Compute A, B, C swizzle modes."""
    var a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    var b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    var c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    if c_type == .bfloat16 or c_type == .float32:
        if AB_swapped:
            c_swizzle = (
                TensorMapSwizzle.SWIZZLE_32B if is_gmm else TensorMapSwizzle.SWIZZLE_128B
            )
        else:
            # When not swapped, output_tile_shape[1] is the N dimension.
            # Key the swizzle off bytes so it is dtype-generic: bf16 tile_n
            # {64,32,16} and fp32 tile_n {32,16,8} both map to {128B,64B,32B}.
            var elem_size = 2 if c_type == DType.bfloat16 else 4
            var row_bytes = output_tile_shape[1] * elem_size
            if row_bytes == 128:
                c_swizzle = TensorMapSwizzle.SWIZZLE_128B
            elif row_bytes == 64:
                c_swizzle = TensorMapSwizzle.SWIZZLE_64B
            elif row_bytes == 32:
                c_swizzle = TensorMapSwizzle.SWIZZLE_32B
    else:
        c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    return (a_swizzle, b_swizzle, c_swizzle)


def _compute_epi_load_swizzle[
    c_type: DType,
](contiguous_elems: Int) -> TensorMapSwizzle:
    """Compute epilogue tensor TMA swizzle mode from the contiguous dimension size.

    Picks the largest swizzle whose byte width evenly divides the row size.
    """
    var row_bytes = contiguous_elems * size_of[c_type]()
    if row_bytes % 128 == 0:
        return TensorMapSwizzle.SWIZZLE_128B
    if row_bytes % 64 == 0:
        return TensorMapSwizzle.SWIZZLE_64B
    if row_bytes % 32 == 0:
        return TensorMapSwizzle.SWIZZLE_32B
    return TensorMapSwizzle.SWIZZLE_NONE


def _maximize_pipeline_stages[
    a_type: DType, b_type: DType, c_type: DType
](
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    output_tile_shape: IndexList[2],
    num_output_stages: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    extra_smem_per_stage: Int = 0,
    use_tma_epilogue_load: Bool = False,
    num_tma_epilogue_pipeline_stages: Int = 2,
    AB_swapped: Bool = False,
    epilogue_is_1d: Bool = False,
) -> Int:
    """Calculate max pipeline stages based on shared memory budget."""
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    var c_smem_bytes = (
        output_tile_shape[0]
        * output_tile_shape[1]
        * num_output_stages
        * size_of[c_type]()
    )
    # Add tmem addr (4 bytes) and tmem dealloc mbar (8 bytes).
    var output_smem_bytes = c_smem_bytes + 12

    var epi_load_smem_bytes = 0
    if use_tma_epilogue_load:
        var num_epi_stages = num_tma_epilogue_pipeline_stages
        if epilogue_is_1d:
            var bias_tile_elems = block_tile_shape[
                0
            ] if AB_swapped else mma_shape[1]
            epi_load_smem_bytes = (
                bias_tile_elems * num_epi_stages * size_of[c_type]()
            )
        elif AB_swapped:
            epi_load_smem_bytes = (
                mma_shape[1]
                * block_tile_shape[0]
                * num_epi_stages
                * size_of[c_type]()
            )
        else:
            epi_load_smem_bytes = (
                block_tile_shape[0]
                * output_tile_shape[1]
                * num_epi_stages
                * size_of[c_type]()
            )
        # Add epi_load barrier pairs × 16 bytes
        epi_load_smem_bytes += num_epi_stages * 16

    # Response 128B, clc mbar 16B, clc-load pipeline mbar 16B.
    var clc_smem_bytes = 160 * num_clc_pipeline_stages
    # Usage by mma-output-pipeline.
    var mma_output_smem_bytes = num_accum_pipeline_stages * 16

    var a_smem_bytes_per_stage = (
        block_tile_shape[0] * block_tile_shape[2] * size_of[a_type]()
    )
    var b_smem_bytes_per_stage = (
        block_tile_shape[1] * block_tile_shape[2] * size_of[b_type]()
    )
    # Include 16 bytes for consumer and producer mbar per stage.
    var AB_smem_per_stage = (
        a_smem_bytes_per_stage
        + b_smem_bytes_per_stage
        + 16
        + extra_smem_per_stage
    )

    return (
        b200_smem
        - output_smem_bytes
        - clc_smem_bytes
        - mma_output_smem_bytes
        - epi_load_smem_bytes
    ) // AB_smem_per_stage


def _write_common_config[
    W: Writer,
    a_type: DType,
    c_type: DType,
    transpose_b: Bool,
](
    mut writer: W,
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    num_pipeline_stages: Int,
    k_group_size: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    output_tile_shape: IndexList[2],
    AB_swapped: Bool,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    c_swizzle: TensorMapSwizzle,
    block_swizzle_size: Int,
    raster_order: RasterOrder,
    num_split_k: Int,
    register_based_epilogue: Bool,
    use_tma_epilogue_load: Bool = False,
    num_tma_epilogue_pipeline_stages: Int = 0,
    epilogue_is_1d: Bool = False,
):
    """Write common config fields to string."""
    writer.write(a_type, "_")
    writer.write(c_type, "_")
    writer.write("cta", cta_group, "_")
    writer.write("mma", mma_shape[0], "x", mma_shape[1], "x", mma_shape[2], "_")
    writer.write(
        "cluster",
        cluster_shape[0],
        "x",
        cluster_shape[1],
        "x",
        cluster_shape[2],
        "_",
    )
    writer.write("stages", num_pipeline_stages, "_")
    writer.write("k_group", k_group_size, "_")
    writer.write("clc", num_clc_pipeline_stages, "_")
    writer.write("accum", num_accum_pipeline_stages, "_")
    writer.write("out", num_output_stages, "_")
    writer.write(output_tile_shape[0], "x", output_tile_shape[1], "_")
    writer.write("swap" if AB_swapped else "noswap", "_")
    writer.write("K_" if transpose_b else "MN_")
    writer.write("asz", a_swizzle.bytes(), "_")
    writer.write("bsz", b_swizzle.bytes(), "_")
    writer.write("csz", c_swizzle.bytes(), "_")
    writer.write("bz", block_swizzle_size, "_", raster_order)
    writer.write("splitk", num_split_k, "_")
    writer.write(
        "rbe_" if register_based_epilogue else "sbe_"
    )  # (rbe) register based epilogue or (sbe) shared memory based epilogue
    if use_tma_epilogue_load:
        writer.write(
            "epi1d_" if epilogue_is_1d else "epi2d_",
            String(num_tma_epilogue_pipeline_stages),
            "stages_",
        )
    writer.write("_")


def _get_dtype_name(dtype: DType) -> String:
    if dtype == .bfloat16:
        return "bf16"
    elif dtype == .float8_e4m3fn:
        return "e4m3"
    elif dtype == .uint8:
        return "e2m1"
    else:
        return String(dtype)


def _get_common_config_string[
    a_type: DType,
    c_type: DType,
    transpose_b: Bool,
](
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    num_pipeline_stages: Int,
    k_group_size: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    num_output_stages: Int,
    output_tile_shape: IndexList[2],
    AB_swapped: Bool,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    c_swizzle: TensorMapSwizzle,
    block_swizzle_size: Int,
    raster_order: RasterOrder,
    num_split_k: Int,
    register_based_epilogue: Bool,
    use_tma_epilogue_load: Bool,
) -> String:
    """Get common config string."""
    return String(
        _get_dtype_name(a_type)
        + "_"
        + _get_dtype_name(a_type)
        + "_"
        + _get_dtype_name(c_type)
        + "_"
        + String(cta_group)
        + "sm_"
        + String(mma_shape[0] // cta_group)
        + "x"
        + String(mma_shape[1] // cta_group)
        + "x"
        + String(mma_shape[2])
        + "_"
        + String(cluster_shape[0])
        + "x"
        + String(cluster_shape[1])
        + "x"
        + String(cluster_shape[2])
        + "_"
        + String(num_pipeline_stages)
        + "stages_"
        + String(k_group_size)
        + "kg_"
        + String(num_clc_pipeline_stages)
        + "clc_"
        + String(num_accum_pipeline_stages)
        + "acc_"
        + String(num_output_stages)
        + "out_"
        + String(output_tile_shape[0])
        + "x"
        + String(output_tile_shape[1])
        + "_"
        + ("swap" if AB_swapped else "noswap")
        + "_"
        + ("K" if transpose_b else "MN")
        + "_"
        + String(a_swizzle.bytes())
        + "asz_"
        + String(b_swizzle.bytes())
        + "bsz_"
        + String(c_swizzle.bytes())
        + "csz_"
        + String(block_swizzle_size)
        + "bz_"
        + String(raster_order)
        + "_"
        + ((String(num_split_k) + "splitk_") if num_split_k > 1 else "")
        + ("rbe_" if register_based_epilogue else "sbe_")
        + ("with_tma_epilogue_" if use_tma_epilogue_load else "")
    )


def _maximize_tma_epi_pipeline_stages[
    a_type: DType, b_type: DType, c_type: DType
](
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    output_tile_shape: IndexList[2],
    num_output_stages: Int,
    num_clc_pipeline_stages: Int,
    num_accum_pipeline_stages: Int,
    max_num_pipeline_stages_wo_tma_epi: Int,
    AB_swapped: Bool,
    epilogue_is_1d: Bool = False,
) -> Int:
    """Calculate max tma epilogue pipeline stages based on smem budget."""

    if AB_swapped or epilogue_is_1d:
        return num_accum_pipeline_stages
    elif mma_shape[0] == mma_shape[1] == 256:
        return 1
    elif max_num_pipeline_stages_wo_tma_epi >= 8:
        return 4
    else:
        return 2


@fieldwise_init
struct MatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Static configuration of GPU matmul.

    Parameters:
        a_type: `DType` of the A (left) operand elements; must equal
            `b_type`.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements; `float32` input
            requires `float32` output.
        transpose_b: Whether the B operand is stored transposed (defaults to
            `True`).
    """

    # Mandatory parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder
    var register_based_epilogue: Bool

    comptime accum_type = get_accum_type[Self.a_type]()

    # Has default values or derivable from mandatory parameters
    var block_tile_shape: IndexList[3]
    var num_split_k: Int
    var num_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_accum_pipeline_stages: Int
    var num_output_stages: Int
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle
    var epi_load_swizzle: TensorMapSwizzle

    var k_group_size: Int
    var prefetch_tiles_n: Int

    var gemm_kind: GEMMKind

    var use_tma_epilogue_load: Bool
    var num_tma_epilogue_pipeline_stages: Int
    var epilogue_is_1d: Bool

    def __init__(
        out self,
        *,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[Self.a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        num_split_k: Int = 1,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: Int = 1,
        prefetch_tiles_n: Int = 0,
        num_pipeline_stages: Optional[Int] = None,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        register_based_epilogue: Bool = True,
        extra_smem_per_stage: Int = 0,
        gemm_kind: GEMMKind = GEMMKind.GEMM,
        use_tma_epilogue_load: Bool = False,
        num_tma_epilogue_pipeline_stages: Optional[Int] = None,
        epilogue_is_1d: Bool = False,
        output_tile_shape: Optional[IndexList[2]] = None,
        c_swizzle: Optional[TensorMapSwizzle] = None,
    ):
        comptime assert Self.a_type == Self.b_type
        comptime assert (
            Self.a_type != .float32 or Self.c_type == .float32
        ), "float32 input only supports float32 output"

        self.cta_group = cta_group
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.k_group_size = k_group_size
        self.prefetch_tiles_n = prefetch_tiles_n
        self.register_based_epilogue = register_based_epilogue
        self.gemm_kind = gemm_kind
        self.use_tma_epilogue_load = use_tma_epilogue_load
        self.epilogue_is_1d = epilogue_is_1d

        self.block_tile_shape = _compute_block_tile_shape[Self.a_type](
            mma_shape, cta_group
        )
        self.output_tile_shape = _compute_output_tile_shape(
            Self.c_type, mma_shape, cta_group, AB_swapped
        )

        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_output_stages = 2
        self.num_split_k = num_split_k

        var swizzles = _compute_swizzle_modes(
            self.c_type, self.output_tile_shape, AB_swapped
        )
        self.a_swizzle = swizzles[0]
        self.b_swizzle = swizzles[1]
        self.c_swizzle = swizzles[2]
        # For non-AB_swapped: each epilogue stage loads BM×stageN (= output_tile_shape[1]).
        # For AB_swapped: tile is MMA_N×BM, contiguous dim is BM.
        var epi_load_contiguous = self.block_tile_shape[
            0
        ] if AB_swapped else self.output_tile_shape[1]
        self.epi_load_swizzle = _compute_epi_load_swizzle[Self.c_type](
            epi_load_contiguous
        )

        # Calculate max pipeline stages without tma epilogue load.
        var max_num_pipeline_stages_wo_tma_epi = _maximize_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.mma_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            extra_smem_per_stage,
            AB_swapped=AB_swapped,
        )

        var max_num_tma_epi_pipeline_stages = _maximize_tma_epi_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.mma_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            max_num_pipeline_stages_wo_tma_epi,
            AB_swapped,
            epilogue_is_1d,
        )

        if num_tma_epilogue_pipeline_stages:
            self.num_tma_epilogue_pipeline_stages = (
                num_tma_epilogue_pipeline_stages.value()
            )
        else:
            self.num_tma_epilogue_pipeline_stages = (
                max_num_tma_epi_pipeline_stages
            )

        var max_num_pipeline_stages = _maximize_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.mma_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            extra_smem_per_stage,
            use_tma_epilogue_load=self.use_tma_epilogue_load,
            num_tma_epilogue_pipeline_stages=self.num_tma_epilogue_pipeline_stages,
            AB_swapped=AB_swapped,
            epilogue_is_1d=self.epilogue_is_1d,
        )

        if num_pipeline_stages:
            assert (
                num_pipeline_stages.value() <= max_num_pipeline_stages
            ), "MatmulConfig requested num_pipeline_stages exceeds smem budget."
            self.num_pipeline_stages = num_pipeline_stages.value()
        else:
            self.num_pipeline_stages = (
                max_num_pipeline_stages if max_num_pipeline_stages <= 16 else 16
            )

        # SM100 kernel only supports k grouping when num_pipeline_stages is a multiple of k_group_size.
        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

        # Optional caller overrides for decode-mode matmul+RS. The fused
        # kernel widens the C SMEM row (output_tile_shape[1]) so per-row
        # TMA slices meet the 128B source-alignment requirement, and forces
        # a non-swizzled C layout so per-row slicing composes. Neither is
        # derivable from mma_shape, so they are explicit opt-in knobs;
        # applied last so the default derivations (block_tile_shape, A/B
        # swizzles) are unaffected.
        if output_tile_shape:
            self.output_tile_shape = output_tile_shape.value()
        if c_swizzle:
            self.c_swizzle = c_swizzle.value()

    def swap_AB_type(
        self,
    ) -> MatmulConfig[Self.b_type, Self.a_type, Self.c_type, Self.transpose_b]:
        return MatmulConfig[
            Self.b_type, Self.a_type, Self.c_type, Self.transpose_b
        ](
            cta_group=self.cta_group,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            AB_swapped=self.AB_swapped,
            num_pipeline_stages=self.num_pipeline_stages,
            num_accum_pipeline_stages=self.num_accum_pipeline_stages,
            num_clc_pipeline_stages=self.num_clc_pipeline_stages,
            block_swizzle_size=self.block_swizzle_size,
            raster_order=self.raster_order,
            k_group_size=self.k_group_size,
            prefetch_tiles_n=self.prefetch_tiles_n,
            num_split_k=self.num_split_k,
            register_based_epilogue=self.register_based_epilogue,
            gemm_kind=self.gemm_kind,
            use_tma_epilogue_load=self.use_tma_epilogue_load,
            num_tma_epilogue_pipeline_stages=self.num_tma_epilogue_pipeline_stages,
            epilogue_is_1d=self.epilogue_is_1d,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("SM100_" + String(self.gemm_kind) + "_")
        _write_common_config[W, Self.a_type, Self.c_type, Self.transpose_b](
            writer,
            self.cta_group,
            self.mma_shape,
            self.cluster_shape,
            self.num_pipeline_stages,
            self.k_group_size,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            self.num_output_stages,
            self.output_tile_shape,
            self.AB_swapped,
            self.a_swizzle,
            self.b_swizzle,
            self.c_swizzle,
            self.block_swizzle_size,
            self.raster_order,
            self.num_split_k,
            self.register_based_epilogue,
            self.use_tma_epilogue_load,
            self.num_tma_epilogue_pipeline_stages,
            self.epilogue_is_1d,
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def get_kernel_name(self) -> String:
        return (
            "SM100_"
            + String(self.gemm_kind)
            + "_"
            + _get_common_config_string[
                Self.a_type, Self.c_type, Self.transpose_b
            ](
                self.cta_group,
                self.mma_shape,
                self.cluster_shape,
                self.num_pipeline_stages,
                self.k_group_size,
                self.num_clc_pipeline_stages,
                self.num_accum_pipeline_stages,
                self.num_output_stages,
                self.output_tile_shape,
                self.AB_swapped,
                self.a_swizzle,
                self.b_swizzle,
                self.c_swizzle,
                self.block_swizzle_size,
                self.raster_order,
                self.num_split_k,
                self.register_based_epilogue,
                self.use_tma_epilogue_load,
            )
        )


def choose_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
    gemm_kind: GEMMKind = GEMMKind.GEMM,
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
](M: Int, N: Int, K: Int, B: Int) -> MatmulConfig[
    a_type, b_type, c_type, transpose_b
]:
    """Select a `MatmulConfig` that minimizes waves per SM for the given problem shape.

    Parameters:
        a_type: `DType` of the A (left) operand elements; must equal
            `b_type`.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        gemm_kind: The `GEMMKind` selecting the kernel variant, for
            example `GEMM` or `BMM` (defaults to `GEMMKind.GEMM`).
        has_epilogue_tensor: Whether the kernel uses a TMA epilogue load
            for an epilogue tensor (defaults to `False`).
        epilogue_is_1d: Whether the epilogue tensor is 1D, for example a
            bias vector (defaults to `False`).

    Args:
        M: The M dimension of the matmul.
        N: The N dimension of the matmul.
        K: The K dimension of the matmul.
        B: The batch dimension of the matmul.

    Returns:
        A `MatmulConfig` tuned for the given problem dimensions.
    """
    comptime assert a_type == b_type, "a_type and b_type must be the same"

    comptime num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    comptime Kbytes_per_mma = 32
    # We use 128B swizzle, tile size in K is 128B over element size.
    comptime BK = 128 // size_of[a_type]()

    comptime M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1  # maybe increased for small M later

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    # Traverse possible combinations of BM x MMA_N to choose the one minimizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    # We first minimize the number of waves, then use the flops to break tie.

    # For small M, swap A and B so that the small M maps to mma_n since it supports
    # a larger range than mma_m.
    if M < M_pivote:
        # when output dtype is float8_e4m3fn, due to output TMA requirement, we need to use 16 as the granularity for 1CTA.
        var MMA_N_GRANULARITY = 16 if c_type == DType.float8_e4m3fn else 8
        for bm, mma_n in product(
            [64, 128],
            range(
                MMA_N_GRANULARITY,
                align_up(M, MMA_N_GRANULARITY) + 1,
                MMA_N_GRANULARITY,
            ),
        ):
            var num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm) * B
            var num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n

    # For large M, use 2xSM mma
    else:

        @__parameter
        @always_inline
        def select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            for bm in [64, 128]:
                var N_aligned = align_up(N, 16)
                var MMA_N_GRANULARITY = 16
                # when output dtype is float8_e4m3fn, due to output TMA requirement, we need to use 32 as the granularity for 2CTA and MMA_M=128.
                if c_type == .float8_e4m3fn and bm == 64:
                    N_aligned = align_up(N, 32)
                    MMA_N_GRANULARITY = 32

                var max_mma_n = min(N_aligned, 256)
                # In practice 64x16 mma creates too many ctas and increase L2
                # load volume, ends up hurting performance.
                var min_mma_n = min(N_aligned, 32)

                for mma_n in range(
                    max_mma_n, min_mma_n - 1, -MMA_N_GRANULARITY
                ):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n) * B
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        # Swap AB may work better for M = 192 and not-multiple-of-128 values.
        # Capture and update min_num_waves, mma_mn
        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    # For small mmas, we group multiple tiles per tma-mma synchronization.
    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    if output_block_size <= 64 * 96 and ceildiv(K, BK) % 2 == 0:
        k_group_size = 2
    # For very small mmas we can group more aggressively.
    if output_block_size <= 64 * 16 and ceildiv(K, BK) % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume per wave.
        # TODO: consider the L2 resue across waves. # spellchecker:disable-line
        var BM = mma_mn[0] // cta_group
        for tile_size in [1, 2, 4, 8]:
            var num_ctas_m = ceildiv(M, BM)
            # When tile_size is small, it's possible that a wave has more ctas
            # then num_ctas_m * tile_size and num_ctas_per_wave_m > num_ctas_m.
            # The ctas mapping will "wrap around" and include following tile_sizes.
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    # TODO: evaluate the comment's perf impact
    # var num_clc_pipeline_stages: Int = Int(min(min_num_waves-1, 2))
    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    var config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=min(2, min_num_waves),
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        k_group_size=k_group_size,
        gemm_kind=gemm_kind,
        use_tma_epilogue_load=has_epilogue_tensor,
        epilogue_is_1d=epilogue_is_1d,
    )

    # At decode M the producer releases its dependents early, so this kernel is
    # resident while the producer still runs and can spend that window on
    # weights. It stops paying once M fills the device. The depth is capped by
    # the ring: a deeper prefetch would fill it before any barrier can fire.
    if M < 128:
        config.prefetch_tiles_n = min(
            4, config.num_pipeline_stages // config.k_group_size
        )

    return config


def build_sm100_matmul_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
    has_epilogue_tensor: Bool = False,
    epilogue_is_1d: Bool = False,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Build a set of `MatmulConfig` instances by sweeping M from 8 to 8192.

    Parameters:
        a_type: `DType` of the A (left) operand elements; must equal
            `b_type`.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        N: The N dimension (output columns) of the matmul.
        K: The K dimension (contraction axis) of the matmul.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        has_epilogue_tensor: Whether the kernel uses a TMA epilogue load
            for an epilogue tensor (defaults to `False`).
        epilogue_is_1d: Whether the epilogue tensor is 1D, for example a
            bias vector (defaults to `False`).

    Returns:
        A set of unique `MatmulConfig` instances covering the swept M range.
    """
    comptime config_t = MatmulConfig[a_type, b_type, c_type, transpose_b]

    var set = Set[config_t]()

    for m in range(8, 256, 8):  # [8, 256)
        var config = choose_config[
            a_type,
            b_type,
            c_type,
            transpose_b,
            has_epilogue_tensor=has_epilogue_tensor,
            epilogue_is_1d=epilogue_is_1d,
        ](m, N, K, 1)
        if config not in set:
            set.add(config)

    for m in range(256, 8192 + 1, 64):  # [256, 8192]
        var config = choose_config[
            a_type,
            b_type,
            c_type,
            transpose_b,
            has_epilogue_tensor=has_epilogue_tensor,
            epilogue_is_1d=epilogue_is_1d,
        ](m, N, K, 1)
        if config not in set:
            set.add(config)

    return set^


def build_sm100_batched_matmul_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
]() -> Set[MatmulConfig[a_type, b_type, c_type, transpose_b]]:
    """Build a set of batched matmul `MatmulConfig` instances by sweeping batch and M.

    Parameters:
        a_type: `DType` of the A (left) operand elements.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        N: The N dimension (output columns) of the matmul.
        K: The K dimension (contraction axis) of the matmul.
        transpose_b: Whether the B operand is stored transposed (defaults to
            `True`).

    Returns:
        A set of unique batched `MatmulConfig` instances covering the swept ranges.
    """
    comptime config_t = MatmulConfig[a_type, b_type, c_type, transpose_b]

    var set = Set[config_t]()

    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        for m in range(8, 256, 8):  # [8, 256)
            var config = choose_config[
                a_type, b_type, c_type, transpose_b, gemm_kind=GEMMKind.BMM
            ](m, N, K, b)
            if config not in set:
                set.add(config)

    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        for m in range(256, 8192 + 1, 64):  # [256, 8192]
            var config = choose_config[
                a_type, b_type, c_type, transpose_b, gemm_kind=GEMMKind.BMM
            ](m, N, K, b)
            if config not in set:
                set.add(config)

    return set^


@fieldwise_init
struct BlockScaledMatmulConfig[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool = True,
](Copyable, Equatable, Hashable, TrivialRegisterPassable, Writable):
    """Static configuration of GPU matmul.

    Parameters:
        a_type: `DType` of the A (left) operand elements; `uint8` indicates
            packed FP4. Must equal `b_type`, except for the W4A8 pair (see
            `block_scaled_operands_compatible`).
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        sfa_dtype: `DType` of the A operand block scaling factors; selects
            the block-scaled MMA kind.
        sfb_dtype: `DType` of the B operand block scaling factors; must
            equal `sfa_dtype`.
        transpose_b: Whether the B operand is stored transposed (defaults to
            `True`).
    """

    # Mandatory parameters
    var cta_group: Int
    var mma_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var AB_swapped: Bool
    var block_swizzle_size: Int
    var raster_order: RasterOrder
    var register_based_epilogue: Bool

    comptime accum_type = get_accum_type[Self.a_type]()

    comptime sf_block_atom_size = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K

    # Has default values or derivable from mandatory parameters
    var block_tile_shape: IndexList[3]
    var num_split_k: Int
    var num_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_accum_pipeline_stages: Int
    var num_output_stages: Int
    var output_tile_shape: IndexList[2]
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle
    var k_group_size: Int
    var scaling_kind: UMMAKind
    var vec_sf_size: Int
    var num_sf_k_tiles: Int
    var is_small_bn: Bool
    var gemm_kind: GEMMKind
    var prefetch_tiles_n: Int

    def __init__(
        out self,
        *,
        scaling_kind: UMMAKind,
        cta_group: Int = 2,
        mma_shape: IndexList[3] = get_mma_shape[Self.a_type, Self.accum_type](),
        cluster_shape: IndexList[3] = Index(2, 1, 1),
        AB_swapped: Bool = False,
        num_split_k: Int = 1,
        block_swizzle_size: Int = 0,
        raster_order: RasterOrder = RasterOrder.AlongM,
        k_group_size: Int = 1,
        num_pipeline_stages: Optional[Int] = None,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        is_gmm: Bool = False,
        is_small_bn: Bool = False,
        register_based_epilogue: Bool = True,
        gemm_kind: GEMMKind = GEMMKind.GEMM,
        prefetch_tiles_n: Int = 0,
    ):
        comptime assert block_scaled_operands_compatible[
            Self.a_type, Self.b_type
        ](), "a_type and b_type must be the same, or the W4A8 pair"

        self.cta_group = cta_group
        self.is_small_bn = is_small_bn
        self.mma_shape = mma_shape
        self.cluster_shape = cluster_shape
        self.AB_swapped = AB_swapped
        self.block_swizzle_size = block_swizzle_size
        self.raster_order = raster_order
        self.k_group_size = k_group_size
        self.register_based_epilogue = register_based_epilogue

        self.block_tile_shape = _compute_block_tile_shape[Self.a_type](
            mma_shape, cta_group
        )
        self.output_tile_shape = _compute_output_tile_shape(
            Self.c_type, mma_shape, cta_group, AB_swapped
        )

        self.gemm_kind = gemm_kind
        self.prefetch_tiles_n = prefetch_tiles_n

        # Scaling factors configuration (SFA, SFB)
        self.scaling_kind = scaling_kind
        if self.scaling_kind == UMMAKind.KIND_MXF4NVF4:
            self.vec_sf_size = NVFP4_SF_VECTOR_SIZE
        elif self.scaling_kind == UMMAKind.KIND_MXF4:
            self.vec_sf_size = MXFP4_SF_VECTOR_SIZE
        else:
            self.vec_sf_size = MXFP8_SF_VECTOR_SIZE
        var sf_k_group_size = self.vec_sf_size * SF_ATOM_K
        # A K-tile spans `block_tile_shape[2]` shared-memory bytes and, under
        # every kind but the FP4-only ones, one element per byte. The FP4-only
        # kinds keep both operands nibble-packed, so their tile covers twice
        # the elements.
        if (
            self.scaling_kind == UMMAKind.KIND_MXF4NVF4
            or self.scaling_kind == UMMAKind.KIND_MXF4
        ):
            self.num_sf_k_tiles = (
                2 * self.block_tile_shape[2]
            ) // sf_k_group_size
        else:
            self.num_sf_k_tiles = self.block_tile_shape[2] // sf_k_group_size

        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_output_stages = 2
        self.num_split_k = num_split_k

        var swizzles = _compute_swizzle_modes(
            self.c_type, self.output_tile_shape, AB_swapped, is_gmm
        )
        self.a_swizzle = swizzles[0]
        self.b_swizzle = swizzles[1]
        self.c_swizzle = swizzles[2]

        # Calculate scaling factor shared memory per stage
        var a_scales_smem_bytes_per_stage = (
            self.num_sf_k_tiles
            * (self.block_tile_shape[0] // SF_MN_GROUP_SIZE)
            * Self.sf_block_atom_size
            * size_of[Self.sfa_dtype]()
        )
        # Always use atom layout size for SFB SMEM — cp.async now scatters
        # into atom positions so tcgen05_cp can bulk-copy to TMEM.
        var b_scales_smem_bytes_per_stage = (
            self.num_sf_k_tiles
            * (
                align_up(self.mma_shape[1], SF_MN_GROUP_SIZE)
                // SF_MN_GROUP_SIZE
            )
            * Self.sf_block_atom_size
            * size_of[Self.sfb_dtype]()
        )

        # right now we only need 8 bytes (one barrier only for producer) but when we separate the sfb tma load and sfb tmem load, we will need 16 bytes.
        var sfb_tmem_load_mbars_size = 16
        var sf_smem_per_stage = (
            a_scales_smem_bytes_per_stage
            + b_scales_smem_bytes_per_stage
            + sfb_tmem_load_mbars_size
        )

        var max_num_pipeline_stages = _maximize_pipeline_stages[
            Self.a_type, Self.b_type, Self.c_type
        ](
            self.block_tile_shape,
            self.mma_shape,
            self.output_tile_shape,
            self.num_output_stages,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            sf_smem_per_stage,
        )

        if num_pipeline_stages:
            assert num_pipeline_stages.value() <= max_num_pipeline_stages, (
                "BlockScaledMatmulConfig requested num_pipeline_stages exceeds"
                " smem budget. "
            )
            self.num_pipeline_stages = num_pipeline_stages.value()
        else:
            self.num_pipeline_stages = max_num_pipeline_stages

        # SM100 kernel only supports k grouping when num_pipeline_stages is a multiple of k_group_size.
        self.num_pipeline_stages = align_down(
            self.num_pipeline_stages, self.k_group_size
        )

    def swap_AB_type(
        self,
    ) -> BlockScaledMatmulConfig[
        Self.b_type,
        Self.a_type,
        Self.c_type,
        Self.sfb_dtype,
        Self.sfa_dtype,
        Self.transpose_b,
    ]:
        return BlockScaledMatmulConfig[
            Self.b_type,
            Self.a_type,
            Self.c_type,
            Self.sfb_dtype,
            Self.sfa_dtype,
            Self.transpose_b,
        ](
            cta_group=self.cta_group,
            mma_shape=self.mma_shape,
            cluster_shape=self.cluster_shape,
            AB_swapped=self.AB_swapped,
            num_pipeline_stages=self.num_pipeline_stages,
            num_accum_pipeline_stages=self.num_accum_pipeline_stages,
            num_clc_pipeline_stages=self.num_clc_pipeline_stages,
            block_swizzle_size=self.block_swizzle_size,
            raster_order=self.raster_order,
            k_group_size=self.k_group_size,
            num_split_k=self.num_split_k,
            scaling_kind=self.scaling_kind,
            is_small_bn=self.is_small_bn,
            register_based_epilogue=self.register_based_epilogue,
            gemm_kind=self.gemm_kind,
            prefetch_tiles_n=self.prefetch_tiles_n,
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("SM100_block_scaled_" + String(self.gemm_kind) + "_")
        writer.write(self.scaling_kind, "_")
        writer.write("A_vec", self.vec_sf_size, "_")
        writer.write(Self.sfa_dtype, "_")
        writer.write("B_vec", self.vec_sf_size, "_")
        writer.write(Self.sfb_dtype, "_")
        writer.write("cpasync_sfb", "_")
        writer.write(
            "small_bn_" if self.is_small_bn else "",
        )
        _write_common_config[W, Self.a_type, Self.c_type, Self.transpose_b](
            writer,
            self.cta_group,
            self.mma_shape,
            self.cluster_shape,
            self.num_pipeline_stages,
            self.k_group_size,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            self.num_output_stages,
            self.output_tile_shape,
            self.AB_swapped,
            self.a_swizzle,
            self.b_swizzle,
            self.c_swizzle,
            self.block_swizzle_size,
            self.raster_order,
            self.num_split_k,
            self.register_based_epilogue,
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def get_kernel_name(self) -> String:
        var name = String("SM100_block_scaled_" + String(self.gemm_kind) + "_")
        name += String(self.scaling_kind) + "_"
        name += _get_common_config_string[
            Self.a_type, Self.c_type, Self.transpose_b
        ](
            self.cta_group,
            self.mma_shape,
            self.cluster_shape,
            self.num_pipeline_stages,
            self.k_group_size,
            self.num_clc_pipeline_stages,
            self.num_accum_pipeline_stages,
            self.num_output_stages,
            self.output_tile_shape,
            self.AB_swapped,
            self.a_swizzle,
            self.b_swizzle,
            self.c_swizzle,
            self.block_swizzle_size,
            self.raster_order,
            self.num_split_k,
            self.register_based_epilogue,
            False,
        )
        name += String("A_vec") + String(self.vec_sf_size) + "_"
        name += String(_get_dtype_name(Self.sfa_dtype)) + "_"
        name += String("B_vec") + String(self.vec_sf_size) + "_"
        name += String(_get_dtype_name(Self.sfb_dtype)) + "_"
        name += String("cpasync_sfb_" if self.is_small_bn else "")

        return name


def choose_block_scaled_config[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool = True,
    gemm_kind: GEMMKind = GEMMKind.GEMM,
](M: Int, N: Int, K: Int) -> BlockScaledMatmulConfig[
    a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
]:
    """Select a `BlockScaledMatmulConfig` that minimizes waves per SM for the given shape.

    Parameters:
        a_type: `DType` of the A (left) operand elements; `uint8`
            indicates packed FP4; must equal `b_type`.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        sfa_dtype: `DType` of the A operand block scaling factors; must
            equal `sfb_dtype`.
        sfb_dtype: `DType` of the B operand block scaling factors; must
            equal `sfa_dtype`.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        gemm_kind: The `GEMMKind` selecting the kernel variant (defaults to
            `GEMMKind.GEMM`).

    Args:
        M: The M dimension of the matmul.
        N: The N dimension of the matmul.
        K: The K dimension of the matmul.

    Returns:
        A `BlockScaledMatmulConfig` tuned for the given problem dimensions.
    """
    comptime assert a_type == b_type, "a_type and b_type must be the same"
    comptime assert (
        sfa_dtype == sfb_dtype
    ), "sfa_dtype and sfb_dtype must be the same"

    comptime is_fp4 = a_type == DType.uint8

    comptime num_SMs = B200.sm_count
    # Nvidia mma instruction process 32B in K.
    comptime Kbytes_per_mma = 32
    # We use 128B swizzle, tile size in K is 128B over element size.
    comptime BK = 128 // size_of[a_type]()

    comptime M_pivote = 32

    var cta_group = 1 if M < M_pivote else 2
    var swapAB = True if M < M_pivote else False
    var k_group_size = 1  # maybe increased for small M later

    var mma_mn = Tuple[Int, Int](256, 256)
    var min_num_waves = Int.MAX

    # Traverse possible combinations of BM x MMA_N to choose the one minimizes the
    # workload per SM. The computation per SM is the flops (ignoring 2x in 2MNK)
    # timed by max number of ctas per SM i.e. number of waves.
    # We first minimize the number of waves, then use the flops to break tie.

    # For small M, swap A and B so that the small M maps to mma_n since it supports
    # a larger range than mma_m.
    if M < M_pivote:
        for bm, mma_n in product([128], range(64, align_up(M, 64) + 1, 64)):
            var num_ctas = ceildiv(M, mma_n) * ceildiv(N, bm)
            var num_waves = ceildiv(num_ctas, num_SMs)
            if num_waves < min_num_waves or (
                num_waves == min_num_waves
                and bm * mma_n < mma_mn[0] * mma_mn[1]
            ):
                min_num_waves = num_waves
                mma_mn[0] = bm
                mma_mn[1] = mma_n

    # For large M, use 2xSM mma
    else:

        @__parameter
        @always_inline
        def select_mma_mn(M: Int, N: Int, _swapAB: Bool = False):
            var N_alignby64 = align_up(N, 64)
            var max_mma_n = min(N_alignby64, 256)
            # In practice 64x16 mma creates too many ctas and increase L2
            # load volume, ends up hurting performance.
            var min_mma_n = min(N_alignby64, 64)
            for bm in [128]:
                for mma_n in range(max_mma_n, min_mma_n - 1, -64):
                    var mma_m = bm * cta_group
                    var num_clusters = ceildiv(M, mma_m) * ceildiv(N, mma_n)
                    var num_waves = ceildiv(num_clusters, num_SMs // cta_group)
                    if num_waves > min_num_waves:
                        break
                    elif num_waves < min_num_waves or (
                        num_waves == min_num_waves
                        and mma_m * mma_n < mma_mn[0] * mma_mn[1]
                    ):
                        min_num_waves = num_waves
                        mma_mn[0] = mma_m
                        mma_mn[1] = mma_n
                        swapAB = _swapAB

        # Swap AB may work better for M = 192 and not-multiple-of-128 values.
        # Capture and update min_num_waves, mma_mn
        select_mma_mn(M, N)
        select_mma_mn(N, M, True)

    # For small mmas, we group multiple tiles per tma-mma synchronization.
    var output_block_size = (mma_mn[0] // cta_group) * mma_mn[1]
    var num_k_iters = ceildiv(K // 2, BK) if is_fp4 else ceildiv(K, BK)
    if output_block_size <= 64 * 96 and num_k_iters % 2 == 0:
        k_group_size = 2
    # For very small mmas we can group more aggressively.
    if output_block_size <= 64 * 16 and num_k_iters % 4 == 0:
        k_group_size = 4

    var min_load_volume = Int.MAX
    var optimal_block_swizzle_size = 0

    # Tile waves when there are >= 4 waves. In theory it should be >=2, but let's
    # be conservative.
    if min_num_waves >= 4:
        # Represent the load volume by
        #    BM * num_ctas_per_wave_m + MMA_N * num_ctas_per_wave_N
        # Use MMA_N because cta_group = 2, 2 ctas cover entire MMA_N. cta_group = 1
        # has BN = MMA_N.
        # Traverse the tile sizes to find min load volume per wave.
        # TODO: consider the L2 resue across waves. # spellchecker:disable-line
        var BM = mma_mn[0] // cta_group
        for tile_size in [1, 2, 4, 8]:
            var num_ctas_m = ceildiv(M, BM)
            # When tile_size is small, it's possible that a wave has more ctas
            # then num_ctas_m * tile_size and num_ctas_per_wave_m > num_ctas_m.
            # The ctas mapping will "wrap around" and include following tile_sizes.
            var num_ctas_per_wave_m = ceildiv(num_SMs, tile_size)
            var num_ctas_per_wave_n = tile_size * ceildiv(
                num_ctas_per_wave_m, num_ctas_m
            )
            num_ctas_per_wave_m = min(num_ctas_per_wave_m, num_ctas_m)
            var load_volume_per_wave = (
                num_ctas_per_wave_m * BM + num_ctas_per_wave_n * mma_mn[1]
            )
            if load_volume_per_wave < min_load_volume:
                min_load_volume = load_volume_per_wave
                optimal_block_swizzle_size = tile_size

    # TODO: evaluate the comment's perf impact
    # var num_clc_pipeline_stages: Int = Int(min(min_num_waves-1, 2))
    var num_clc_pipeline_stages = 0 if min_num_waves == 1 else 2

    var num_accum_pipeline_stages = 2 if mma_mn[1] <= 128 else 1

    # `a_type == b_type` is asserted above, so this can never see the mixed
    # W4A8 pair -- it reads B's dtype only to stay on the shared mapping.
    comptime scaling_kind = block_scaled_umma_kind[a_type, b_type, sfa_dtype]()

    return BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        mma_shape=IndexList[3](
            mma_mn[0], mma_mn[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=swapAB,
        block_swizzle_size=optimal_block_swizzle_size,
        num_accum_pipeline_stages=num_accum_pipeline_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        k_group_size=k_group_size,
        gemm_kind=gemm_kind,
    )


def build_block_scaled_configs[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    N: Int,
    K: Int,
    transpose_b: Bool = True,
]() -> Set[
    BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]
]:
    """Build a set of `BlockScaledMatmulConfig` instances by sweeping M from 8 to 8192.

    Parameters:
        a_type: `DType` of the A (left) operand elements.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        sfa_dtype: `DType` of the A operand block scaling factors.
        sfb_dtype: `DType` of the B operand block scaling factors.
        N: The N dimension (output columns) of the matmul.
        K: The K dimension (contraction axis) of the matmul.
        transpose_b: Whether the B operand is stored transposed (defaults to
            `True`).

    Returns:
        A set of unique `BlockScaledMatmulConfig` instances covering the swept M range.
    """
    comptime config_t = BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ]

    var set = Set[config_t]()

    # Enumerate only MMA tiles the kernel accepts (its tile `constrained[]`):
    # one unsupported tile in this comptime-instantiated set fails the whole
    # compile; a shape with no matching config falls through to vendor.
    def _kernel_supported(cfg: config_t) {} -> Bool:
        var mma_m_ok = cfg.mma_shape[0] == (256 if cfg.cta_group == 2 else 128)
        return mma_m_ok and cfg.mma_shape[1] in (64, 128, 192, 256)

    for m in range(8, 128, 8):  # [8, 128]
        var config = choose_block_scaled_config[
            a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
        ](m, N, K)
        if _kernel_supported(config) and config not in set:
            set.add(config)

    for m in range(128, 8193, 64):  # [128, 8192]
        var config = choose_block_scaled_config[
            a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
        ](m, N, K)
        if _kernel_supported(config) and config not in set:
            set.add(config)

    return set^


def default_matmul_config_bf16_fp8[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool = True,
    cta_group: Int = 2,
    gemm_kind: GEMMKind = GEMMKind.GEMM,
    has_epilogue_tensor: Bool = False,
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b]:
    """Return a default `MatmulConfig` for bf16-output FP8 matmul kernels.

    Parameters:
        a_type: `DType` of the A (left) operand elements; must equal
            `b_type`.
        b_type: `DType` of the B (right) operand elements.
        c_type: `DType` of the output matrix elements.
        transpose_b: Whether the B operand is stored transposed (defaults
            to `True`).
        cta_group: CTA group size, 1 or 2, setting the number of CTAs
            cooperating per MMA (defaults to 2).
        gemm_kind: The `GEMMKind` selecting the kernel variant (defaults
            to `GEMMKind.GEMM`).
        has_epilogue_tensor: Whether the kernel uses a TMA epilogue load
            for an epilogue tensor (defaults to `False`).

    Returns:
        A `MatmulConfig` with a 128x128 block tile and default pipeline stages.
    """
    # Nvidia mma instruction process 32B in K.
    comptime Kbytes_per_mma = 32

    comptime MMA_K = 32 // size_of[a_type]()
    comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]()

    comptime block_tile_shape = Index(128, 128, BK)
    comptime umma_shape = Index(
        block_tile_shape[0] * cta_group, block_tile_shape[1] * cta_group, MMA_K
    )

    return MatmulConfig[a_type, b_type, c_type, transpose_b](
        mma_shape=IndexList[3](
            umma_shape[0], umma_shape[1], Kbytes_per_mma // size_of[a_type]()
        ),
        cta_group=cta_group,
        cluster_shape=Index(cta_group, 1, 1),
        AB_swapped=False,
        block_swizzle_size=0,
        num_accum_pipeline_stages=2,
        num_clc_pipeline_stages=2,
        k_group_size=1,
        gemm_kind=gemm_kind,
        use_tma_epilogue_load=has_epilogue_tensor,
    )
