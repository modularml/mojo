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

"""Defines tuning configurations for SM100 structured matmul kernels.

Holds the `TuningConfigSM100` and `TuningConfigSmallMNGemms` structs that
bundle kernel launch parameters for specific MxNxK matmul shapes, along with
the curated tuning lists used by the SM100 structured dispatch tables.
"""

from ...tile_scheduler import RasterOrder
from linalg.gemv import GEMVAlgorithm
from internal_utils import TuningConfig
from std.utils.index import Index, IndexList


struct TuningConfigSM100(TrivialRegisterPassable, TuningConfig):
    """Holds SM100 matmul kernel launch parameters for a range of MxNxK shapes.

    Stores the MMA shape, block tile shape, cluster shape, swizzle and
    rasterization settings, pipeline stage counts, and split-K factor that
    select an optimized kernel configuration for matmuls whose M dimension
    falls in the half-open interval `[M, M_end)`.
    """

    # The kernel parameters are optimal for shape in [M:M_end]xNxK.
    var M: Int
    var M_end: Int
    var N: Int
    var K: Int

    # Kernel parameters
    var mma_shape: IndexList[3]
    var block_tile_shape: IndexList[3]
    var cluster_shape: IndexList[3]
    var block_swizzle_size: Int
    var rasterize_order: RasterOrder
    var cta_group: Int
    var swapAB: Bool
    var k_group_size: Int
    var num_accum_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var num_split_k: Int
    var num_pipeline_stages: Int  # 0 = auto-compute
    var is_small_bn: Bool

    var batch_size: Int

    def __init__(
        out self,
        M: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        block_tile_shape: IndexList[3],
        cluster_shape: IndexList[3],
        block_swizzle_size: Int,
        rasterize_order: RasterOrder,
        cta_group: Int = 2,
        swapAB: Bool = False,
        k_group_size: Int = 1,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        num_split_k: Int = 1,
        num_pipeline_stages: Int = 0,
        is_small_bn: Bool = False,
        batch_size: Int = 1,
    ):
        self.M = M
        self.M_end = M + 1
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.block_tile_shape = block_tile_shape
        self.cluster_shape = cluster_shape
        self.block_swizzle_size = block_swizzle_size
        self.rasterize_order = rasterize_order
        self.cta_group = cta_group
        self.swapAB = swapAB
        self.k_group_size = k_group_size
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_split_k = num_split_k
        self.num_pipeline_stages = num_pipeline_stages
        self.is_small_bn = is_small_bn
        self.batch_size = batch_size

    def write_to(self, mut writer: Some[Writer]):
        """Writes the tuning config as a string.

        Args:
            writer: The writer to write to.
        """
        writer.write(
            "config: ",
            "m:",
            self.M,
            "/n:",
            self.N,
            "/k:",
            self.K,
            "/b:",
            self.batch_size,
        )

    def __init__(
        out self,
        M: Int,
        M_end: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        cta_group: Int,
        cluster_shape: IndexList[3],
        block_swizzle_size: Int,
        rasterize_order: RasterOrder,
        swapAB: Bool = False,
        k_group_size: Int = 1,
        num_accum_pipeline_stages: Int = 2,
        num_clc_pipeline_stages: Int = 2,
        num_split_k: Int = 1,
        num_pipeline_stages: Int = 0,
        is_small_bn: Bool = False,
        batch_size: Int = 1,
    ):
        self.M = M
        self.M_end = M_end
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.cta_group = cta_group
        self.block_tile_shape = Index(
            mma_shape[0] // cta_group,
            mma_shape[1] // cta_group,
            mma_shape[2] * 4,
        )
        self.cluster_shape = cluster_shape
        self.block_swizzle_size = block_swizzle_size
        self.rasterize_order = rasterize_order
        self.swapAB = swapAB
        self.k_group_size = k_group_size
        self.num_accum_pipeline_stages = num_accum_pipeline_stages
        self.num_clc_pipeline_stages = num_clc_pipeline_stages
        self.num_split_k = num_split_k
        self.num_pipeline_stages = num_pipeline_stages
        self.is_small_bn = is_small_bn
        self.batch_size = batch_size


struct TuningConfigSmallMNGemms(TrivialRegisterPassable, TuningConfig):
    """Holds launch parameters for small-M, small-N GEMM/GEMV kernels.

    Stores the tile dimensions, thread count, unroll factor, K-tile size,
    and GEMV algorithm kind that select an optimized kernel configuration for
    matmuls whose M dimension falls in the half-open interval `[M, M_end)`.
    """

    var M: Int
    var M_end: Int
    var N: Int
    var K: Int
    var kernel_kind: GEMVAlgorithm
    var tile_m: Int
    var tile_n: Int
    var num_threads: Int
    var unroll_factor: Int
    var tile_k: Int
    var swapAB: Bool

    def __init__(
        out self,
        M: Int,
        M_end: Int,
        N: Int,
        K: Int,
        tile_m: Int,
        tile_n: Int,
        num_threads: Int,
        kernel_kind: GEMVAlgorithm,
        unroll_factor: Int = 1,
        tile_k: Int = 128,
        swapAB: Bool = False,
    ):
        self.M = M
        self.M_end = M_end
        self.N = N
        self.K = K
        self.kernel_kind = kernel_kind
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.unroll_factor = unroll_factor
        self.tile_k = tile_k
        self.swapAB = swapAB

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "small_mn_config: ",
            "kernel:",
            self.kernel_kind,
            "/m:",
            self.M,
            "-",
            self.M_end,
            "/n:",
            self.N,
            "/k:",
            self.K,
            "/tile_m:",
            self.tile_m,
            "/tile_n:",
            self.tile_n,
            "/threads:",
            self.num_threads,
            "/swapAB:",
            self.swapAB,
        )


# codegen template
# TuningConfigSM100(
#     M=[@M],
#     N=[@N],
#     K=[@K],
#     mma_shape=Index([@TUNE_BM] * 2, [@TUNE_BN] * 2, mma_k),
#     block_tile_shape=Index([@TUNE_BM], [@TUNE_BN], bk),
#     cluster_shape=Index([@TUNE_CLUSTER_DIM_X], [@TUNE_CLUSTER_DIM_Y], [@TUNE_CLUSTER_DIM_Z]),
#     block_swizzle_size=[@TUNE_BLOCK_SWIZZLE_SIZE],
#     rasterize_order=RasterOrder([@TUNE_RASTER_ORDER]),
# )

# ===----------------------------------------------------------------------=== #
# BF16 outliers
# ===----------------------------------------------------------------------=== #


def _get_tuning_list_sm100_bf16() -> List[TuningConfigSM100]:
    return [
        # ----------------BEGIN-TUNING-LIST-SM100-BF16----------------
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [0]
        TuningConfigSM100(
            M=256,
            M_end=320,
            N=20480,
            K=5376,
            mma_shape=Index(128, 224, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [1]
        TuningConfigSM100(
            M=256,
            M_end=320,
            N=16384,
            K=5376,
            mma_shape=Index(128, 224, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [2]
        TuningConfigSM100(
            M=256,
            M_end=320,
            N=43008,
            K=5376,
            mma_shape=Index(128, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [3]
        TuningConfigSM100(
            M=8192,
            M_end=131136,
            N=1536,
            K=4096,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [4]
        TuningConfigSM100(
            M=8192,
            M_end=131136,
            N=1536,
            K=1536,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [5]
        TuningConfigSM100(
            M=8192,
            M_end=131136,
            N=4608,
            K=1536,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [6]
        TuningConfigSM100(
            M=4096,
            M_end=4160,
            N=1024,
            K=512,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(0),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [7]
        TuningConfigSM100(
            M=4992,
            M_end=5184,
            N=1024,
            K=512,
            mma_shape=Index(256, 160, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [8]
        TuningConfigSM100(
            M=25,
            M_end=32,
            N=7168,
            K=1024,
            mma_shape=Index(256, 32, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(0),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [9]
        TuningConfigSM100(
            M=2048,
            M_end=2112,
            N=1536,
            K=1536,
            mma_shape=Index(256, 192, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(0),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [10]
        TuningConfigSM100(
            M=32,
            M_end=33,
            N=1536,
            K=1536,
            mma_shape=Index(64, 8, 16),
            cta_group=1,
            cluster_shape=Index(2, 4, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(0),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [11]
        TuningConfigSM100(
            M=2048,
            M_end=2112,
            N=16384,
            K=512,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(0),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [12]
        TuningConfigSM100(
            M=2112,
            M_end=74432,
            N=16384,
            K=512,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [13]
        TuningConfigSM100(
            M=3456,
            M_end=3520,
            N=43008,
            K=5376,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [14]
        TuningConfigSM100(
            M=48000,
            M_end=48064,
            N=5376,
            K=21504,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [15]
        TuningConfigSM100(
            M=87,
            M_end=129,
            N=3072,
            K=4096,
            mma_shape=Index(128, 48, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [16]
        TuningConfigSM100(
            M=449,
            M_end=513,
            N=3072,
            K=4096,
            mma_shape=Index(128, 192, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [17]
        TuningConfigSM100(
            M=9,
            M_end=32,
            N=4096,
            K=7168,
            mma_shape=Index(128, 16, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [18]
        TuningConfigSM100(
            M=65,
            M_end=81,
            N=6144,
            K=4096,
            mma_shape=Index(128, 80, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [19]
        TuningConfigSM100(
            M=81,
            M_end=97,
            N=6144,
            K=4096,
            mma_shape=Index(128, 96, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [20]
        TuningConfigSM100(
            M=9,
            M_end=32,
            N=6144,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [21]
        TuningConfigSM100(
            M=9,
            M_end=17,
            N=4096,
            K=4096,
            mma_shape=Index(128, 16, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [22]
        TuningConfigSM100(
            M=17,
            M_end=64,
            N=4096,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [23]
        TuningConfigSM100(
            M=9,
            M_end=22,
            N=28672,
            K=4096,
            mma_shape=Index(128, 32, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [24]
        TuningConfigSM100(
            M=23,
            M_end=47,
            N=28672,
            K=4096,
            mma_shape=Index(128, 48, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [25]
        TuningConfigSM100(
            M=48,
            M_end=65,
            N=28672,
            K=4096,
            mma_shape=Index(256, 64, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [26]
        TuningConfigSM100(
            M=65,
            M_end=81,
            N=28672,
            K=4096,
            mma_shape=Index(256, 80, 16),
            cta_group=2,
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [27]
        TuningConfigSM100(
            M=4608,
            M_end=4672,
            N=6144,
            K=24576,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [28]
        TuningConfigSM100(
            M=4608,
            M_end=4672,
            N=6144,
            K=18432,
            mma_shape=Index(256, 256, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_bf16.yaml]
        # index: [29]
        TuningConfigSM100(
            M=12,
            M_end=65,
            N=20480,
            K=7168,
            mma_shape=Index(128, 160, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=False,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # ----------------END-TUNING-LIST-SM100-BF16----------------
    ]


# ===----------------------------------------------------------------------=== #
# FP32 Shapes
# ===----------------------------------------------------------------------=== #


def _get_tuning_list_sm100_fp32() -> List[TuningConfigSM100]:
    return List[TuningConfigSM100]()


# ===----------------------------------------------------------------------=== #
# FP8 Shapes
# ===----------------------------------------------------------------------=== #


def _get_tuning_list_sm100_fp8[
    mma_k: Int, bk: Int
]() -> List[TuningConfigSM100]:
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-FP8----------------
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [0]
        TuningConfigSM100(
            M=150,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [1]
        TuningConfigSM100(
            M=225,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(4, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [2]
        TuningConfigSM100(
            M=256,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [3]
        TuningConfigSM100(
            M=300,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [4]
        TuningConfigSM100(
            M=450,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [5]
        TuningConfigSM100(
            M=512,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [6]
        TuningConfigSM100(
            M=600,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [7]
        TuningConfigSM100(
            M=750,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 48 * 2, mma_k),
            block_tile_shape=Index(128, 48, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [8]
        TuningConfigSM100(
            M=768,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [9]
        TuningConfigSM100(
            M=1024,
            N=2304,
            K=16384,
            mma_shape=Index(64 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(64, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [10]
        TuningConfigSM100(
            M=2048,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [11]
        TuningConfigSM100(
            M=4096,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [12]
        TuningConfigSM100(
            M=6144,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [13]
        TuningConfigSM100(
            M=8192,
            N=2304,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [14]
        TuningConfigSM100(
            M=150,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [15]
        TuningConfigSM100(
            M=225,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(64, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [16]
        TuningConfigSM100(
            M=300,
            N=4608,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [17]
        TuningConfigSM100(
            M=450,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 64 * 2, mma_k),
            block_tile_shape=Index(128, 64, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [18]
        TuningConfigSM100(
            M=600,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [19]
        TuningConfigSM100(
            M=750,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [20]
        TuningConfigSM100(
            M=2048,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [21]
        TuningConfigSM100(
            M=4096,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [22]
        TuningConfigSM100(
            M=6144,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [23]
        TuningConfigSM100(
            M=8192,
            N=4608,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [24]
        TuningConfigSM100(
            M=150,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [25]
        TuningConfigSM100(
            M=225,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [26]
        TuningConfigSM100(
            M=256,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [27]
        TuningConfigSM100(
            M=300,
            N=13312,
            K=16384,
            mma_shape=Index(64 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(64, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [28]
        TuningConfigSM100(
            M=450,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [29]
        TuningConfigSM100(
            M=512,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [30]
        TuningConfigSM100(
            M=600,
            N=13312,
            K=16384,
            mma_shape=Index(64 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(64, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [31]
        TuningConfigSM100(
            M=750,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [32]
        TuningConfigSM100(
            M=768,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [33]
        TuningConfigSM100(
            M=1024,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [34]
        TuningConfigSM100(
            M=2048,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 104 * 2, mma_k),
            block_tile_shape=Index(128, 104, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [35]
        TuningConfigSM100(
            M=4096,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [36]
        TuningConfigSM100(
            M=6144,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [37]
        TuningConfigSM100(
            M=8192,
            N=13312,
            K=16384,
            mma_shape=Index(128 * 2, 88 * 2, mma_k),
            block_tile_shape=Index(128, 88, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [38]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [39]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [40]
        TuningConfigSM100(
            M=256,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [41]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [42]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [43]
        TuningConfigSM100(
            M=512,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [44]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=2048,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [45]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [46]
        TuningConfigSM100(
            M=768,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=2,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [47]
        TuningConfigSM100(
            M=1024,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [48]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [49]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [50]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [51]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=2048,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [52]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [53]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [54]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=4096,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [55]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [56]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=4096,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [57]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [58]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [59]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [60]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [61]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=4096,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [62]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [63]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [64]
        TuningConfigSM100(
            M=256,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [65]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=6656,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [66]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [67]
        TuningConfigSM100(
            M=512,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [68]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=6656,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [69]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [70]
        TuningConfigSM100(
            M=768,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [71]
        TuningConfigSM100(
            M=1024,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [72]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [73]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [74]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [75]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=6656,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [76]
        TuningConfigSM100(
            M=150,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [77]
        TuningConfigSM100(
            M=225,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [78]
        TuningConfigSM100(
            M=300,
            N=16384,
            K=13312,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [79]
        TuningConfigSM100(
            M=450,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [80]
        TuningConfigSM100(
            M=600,
            N=16384,
            K=13312,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [81]
        TuningConfigSM100(
            M=750,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [82]
        TuningConfigSM100(
            M=2048,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [83]
        TuningConfigSM100(
            M=4096,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [84]
        TuningConfigSM100(
            M=6144,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [85]
        TuningConfigSM100(
            M=8192,
            N=16384,
            K=13312,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [86]
        TuningConfigSM100(
            M=150,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [87]
        TuningConfigSM100(
            M=225,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 96 * 2, mma_k),
            block_tile_shape=Index(128, 96, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [88]
        TuningConfigSM100(
            M=300,
            N=26624,
            K=16384,
            mma_shape=Index(64 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(64, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [89]
        TuningConfigSM100(
            M=450,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [90]
        TuningConfigSM100(
            M=600,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [91]
        TuningConfigSM100(
            M=750,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 112 * 2, mma_k),
            block_tile_shape=Index(128, 112, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [92]
        TuningConfigSM100(
            M=2048,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [93]
        TuningConfigSM100(
            M=4096,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 104 * 2, mma_k),
            block_tile_shape=Index(128, 104, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [94]
        TuningConfigSM100(
            M=6144,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 120 * 2, mma_k),
            block_tile_shape=Index(128, 120, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=4,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [95]
        TuningConfigSM100(
            M=8192,
            N=26624,
            K=16384,
            mma_shape=Index(128 * 2, 128 * 2, mma_k),
            block_tile_shape=Index(128, 128, bk),
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [96]
        TuningConfigSM100(
            M=7000,
            M_end=7000 + 32,
            N=43008,
            K=5376,
            mma_shape=Index(256, 256, mma_k),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
        ),
        # Automatically generated from [tuning_table_sm100_fp8.yaml]
        # index: [97]
        TuningConfigSM100(
            M=256,
            M_end=256 + 32,
            N=8192,
            K=5376,
            mma_shape=Index(256, 128, mma_k),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=1,
            rasterize_order=RasterOrder(1),
        ),
        # ----------------END-TUNING-LIST-SM100-FP8----------------
    ]

    return materialize[config_list]()


def _get_tuning_list_sm100_nvfp4() -> List[TuningConfigSM100]:
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-NVFP4----------------
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [0]
        TuningConfigSM100(
            M=1,
            M_end=17,
            N=18432,
            K=7168,
            mma_shape=Index(256, 16, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [1]
        TuningConfigSM100(
            M=17,
            M_end=33,
            N=18432,
            K=7168,
            mma_shape=Index(256, 32, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [2]
        TuningConfigSM100(
            M=32,
            M_end=129,
            N=4096,
            K=7168,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [3]
        TuningConfigSM100(
            M=65,
            M_end=129,
            N=7168,
            K=8192,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [4]
        TuningConfigSM100(
            M=2,
            M_end=32,
            N=7168,
            K=18432,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [5]
        TuningConfigSM100(
            M=64,
            M_end=129,
            N=7168,
            K=18432,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 2, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [6]
        TuningConfigSM100(
            M=1,
            M_end=9,
            N=36864,
            K=7168,
            mma_shape=Index(128, 8, 32),
            cta_group=1,
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=8,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [7]
        TuningConfigSM100(
            M=9,
            M_end=17,
            N=36864,
            K=7168,
            mma_shape=Index(128, 16, 32),
            cta_group=1,
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=8,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [8]
        TuningConfigSM100(
            M=17,
            M_end=25,
            N=36864,
            K=7168,
            mma_shape=Index(128, 24, 32),
            cta_group=1,
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=8,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [9]
        TuningConfigSM100(
            M=25,
            M_end=33,
            N=36864,
            K=7168,
            mma_shape=Index(128, 32, 32),
            cta_group=1,
            cluster_shape=Index(1, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            num_split_k=1,
            num_pipeline_stages=8,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [10]
        TuningConfigSM100(
            M=33,
            M_end=65,
            N=18432,
            K=7168,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=True,
            batch_size=1,
        ),
        # Automatically generated from [tuning_table_sm100_nvfp4.yaml]
        # index: [11]
        TuningConfigSM100(
            M=65,
            M_end=69,
            N=18432,
            K=7168,
            mma_shape=Index(256, 96, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=8,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=True,
            batch_size=1,
        ),
        # ----------------END-TUNING-LIST-SM100-NVFP4----------------
    ]

    return materialize[config_list]()


def _get_tuning_list_sm100_mxfp4() -> List[TuningConfigSM100]:
    # MXFP4 uses SF_VEC=32 like MXFP8 and KIND_MXF4 at the hardware level.
    # Start with MXFP8 tuning configs; tune later.
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-MXFP4----------------
        # Automatically generated from [tuning_table_sm100_mxfp4.yaml]
        # index: [0]
        TuningConfigSM100(
            M=1,
            M_end=2,
            N=7168,
            K=16384,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # ----------------END-TUNING-LIST-SM100-MXFP4----------------
    ]

    return materialize[config_list]()


def _get_tuning_list_sm100_mxfp8() -> List[TuningConfigSM100]:
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-MXFP8----------------
        # Automatically generated from [tuning_table_sm100_mxfp8.yaml]
        # index: [0]
        TuningConfigSM100(
            M=1,
            M_end=2,
            N=7168,
            K=16384,
            mma_shape=Index(256, 64, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=1,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=1,
        ),
        # ----------------END-TUNING-LIST-SM100-MXFP8----------------
    ]

    return materialize[config_list]()


def _get_tuning_list_sm100_batched_bf16() -> List[TuningConfigSM100]:
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-BATCHED-BF16----------------
        # Automatically generated from [tuning_table_sm100_batched_bf16.yaml]
        # index: [0]
        TuningConfigSM100(
            M=1,
            M_end=17,
            N=128,
            K=512,
            mma_shape=Index(128, 16, 16),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=64,
        ),
        # ----------------END-TUNING-LIST-SM100-BATCHED-BF16----------------
    ]

    return materialize[config_list]()


def _get_tuning_list_sm100_batched_fp8() -> List[TuningConfigSM100]:
    comptime config_list: List[TuningConfigSM100] = [
        # ----------------BEGIN-TUNING-LIST-SM100-BATCHED-FP8----------------
        # Automatically generated from [tuning_table_sm100_batched_fp8.yaml]
        # index: [0]
        TuningConfigSM100(
            M=1,
            M_end=17,
            N=128,
            K=512,
            mma_shape=Index(128, 16, 32),
            cta_group=2,
            cluster_shape=Index(2, 1, 1),
            block_swizzle_size=0,
            rasterize_order=RasterOrder(1),
            swapAB=True,
            k_group_size=4,
            num_accum_pipeline_stages=1,
            num_clc_pipeline_stages=0,
            num_split_k=1,
            num_pipeline_stages=0,
            is_small_bn=False,
            batch_size=64,
        ),
        # ----------------END-TUNING-LIST-SM100-BATCHED-FP8----------------
    ]

    return materialize[config_list]()


# ===----------------------------------------------------------------------=== #
# Batched FP32 Shapes
# ===----------------------------------------------------------------------=== #


def _get_tuning_list_sm100_batched_fp32() -> List[TuningConfigSM100]:
    return List[TuningConfigSM100]()


# ===----------------------------------------------------------------------=== #
# GEMV tuning configs for small-N shapes
# ===----------------------------------------------------------------------=== #


def _get_tuning_list_small_MN_gemms_bf16() -> List[TuningConfigSmallMNGemms]:
    return [
        # ----------------BEGIN-TUNING-LIST-SM100-SMALL-MN-BF16----------------
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [0]
        TuningConfigSmallMNGemms(
            M=2,
            M_end=5,
            N=384,
            K=7168,
            tile_m=1,
            tile_n=2,
            num_threads=256,
            kernel_kind=GEMVAlgorithm.GEMV_SPLIT_K,
            unroll_factor=1,
            tile_k=128,
            swapAB=False,
        ),
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [1]
        TuningConfigSmallMNGemms(
            M=5,
            M_end=9,
            N=384,
            K=7168,
            tile_m=2,
            tile_n=2,
            num_threads=128,
            kernel_kind=GEMVAlgorithm.GEMV_SPLIT_K,
            unroll_factor=2,
            tile_k=128,
            swapAB=False,
        ),
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [2]
        TuningConfigSmallMNGemms(
            M=9,
            M_end=13,
            N=384,
            K=7168,
            tile_m=2,
            tile_n=2,
            num_threads=128,
            kernel_kind=GEMVAlgorithm.GEMV_SPLIT_K,
            unroll_factor=2,
            tile_k=128,
            swapAB=False,
        ),
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [3]
        TuningConfigSmallMNGemms(
            M=13,
            M_end=17,
            N=384,
            K=7168,
            tile_m=2,
            tile_n=2,
            num_threads=128,
            kernel_kind=GEMVAlgorithm.GEMV_SPLIT_K,
            unroll_factor=1,
            tile_k=128,
            swapAB=False,
        ),
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [4]
        TuningConfigSmallMNGemms(
            M=17,
            M_end=25,
            N=384,
            K=7168,
            tile_m=4,
            tile_n=2,
            num_threads=128,
            kernel_kind=GEMVAlgorithm.GEMV_SPLIT_K,
            unroll_factor=2,
            tile_k=128,
            swapAB=False,
        ),
        # Automatically generated from [tuning_table_sm100_small_mn_bf16.yaml]
        # index: [5]
        TuningConfigSmallMNGemms(
            M=25,
            M_end=33,
            N=384,
            K=7168,
            tile_m=16,
            tile_n=8,
            num_threads=256,
            kernel_kind=GEMVAlgorithm.GEMM_MMA_CPASYNC,
            unroll_factor=1,
            tile_k=256,
            swapAB=False,
        ),
        # ----------------END-TUNING-LIST-SM100-SMALL-MN-BF16----------------
    ]
