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

"""
Defines the SM90 (Hopper) matmul tuning configuration record and the
pre-tuned BF16 lookup table of kernel hyperparameters for specific
(M, N, K) problem shapes.
"""

from ..tile_scheduler import MatmulSchedule, RasterOrder
from internal_utils import TuningConfig
from std.utils.index import Index, IndexList
from std.collections import OptionalReg
from max.gpu.host.info import H100


@fieldwise_init
struct TuningGroup(Equatable, TrivialRegisterPassable, Writable):
    var _value: Int32

    comptime CORE = Self(0)
    comptime MISCELLANEOUS = Self(1)
    comptime INTERNVL = Self(2)
    comptime LLAMA_3_3_70B = Self(3)
    comptime GEMMA_3_27B = Self(4)


struct TuningConfigSM90(TrivialRegisterPassable, TuningConfig):
    """Tuning record for a single (M, N, K) SM90 matmul configuration.

    Stores the full set of kernel hyperparameters: MMA shape, block tile
    shape, pipeline stages, cluster shape, consumer count, multicast flag,
    optional grid shape, scheduling strategy, split-K count, and raster
    order, corresponding to one entry in the pre-tuned BF16 lookup table.
    """

    var M: Int
    var N: Int
    var K: Int

    var mma_shape: IndexList[3]
    var block_tile_shape: IndexList[3]
    var num_pipeline_stages: Int
    var cluster_shape: IndexList[3]
    var num_consumer: Int
    var partitioned_multicast: Bool
    var grid_shape: OptionalReg[IndexList[2]]  # = None
    var schedule: MatmulSchedule  # =  MatmulSchedule.NONE
    var splits: OptionalReg[Int]
    var raster_order: OptionalReg[RasterOrder]
    var dispatch_group: TuningGroup

    def __init__(
        out self,
        M: Int,
        N: Int,
        K: Int,
        mma_shape: IndexList[3],
        block_tile_shape: IndexList[3],
        num_pipeline_stages: Int,
        cluster_shape: IndexList[3],
        num_consumer: Int,
        partitioned_multicast: Bool,
        grid_shape: OptionalReg[IndexList[2]] = None,
        schedule: MatmulSchedule = MatmulSchedule.NONE,
        splits: OptionalReg[Int] = None,
        raster_order: OptionalReg[RasterOrder] = None,
        dispatch_group: TuningGroup = TuningGroup.CORE,
    ):
        self.M = M
        self.N = N
        self.K = K
        self.mma_shape = mma_shape
        self.block_tile_shape = block_tile_shape
        self.num_pipeline_stages = num_pipeline_stages
        self.cluster_shape = cluster_shape
        self.num_consumer = num_consumer
        self.partitioned_multicast = partitioned_multicast
        self.grid_shape = grid_shape
        self.schedule = schedule
        self.splits = splits
        self.raster_order = raster_order
        self.dispatch_group = dispatch_group

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
            "/group:",
            self.dispatch_group,
        )


def _get_tuning_list_bf16[
    size_factor: Int, mma_k: Int, BK: Int
]() -> List[TuningConfigSM90]:
    # kprofile -s oss/modular/max/kernels/src/linalg/matmul/gpu/sm90/tuning.mojo.snippet oss/modular/max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml
    comptime config_list: List[TuningConfigSM90] = [
        # ----------------BEGIN-TUNING-LIST-BF16----------------
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [0]
        TuningConfigSM90(
            M=1,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [1]
        TuningConfigSM90(
            M=2,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [2]
        TuningConfigSM90(
            M=4,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [3]
        TuningConfigSM90(
            M=8,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [4]
        TuningConfigSM90(
            M=16,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [5]
        TuningConfigSM90(
            M=32,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [6]
        TuningConfigSM90(
            M=48,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [7]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [8]
        TuningConfigSM90(
            M=80,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [9]
        TuningConfigSM90(
            M=96,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [10]
        TuningConfigSM90(
            M=112,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 2, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [11]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 2, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [12]
        TuningConfigSM90(
            M=192,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 160, mma_k),
            block_tile_shape=Index(64 * 1, 160, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [13]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=2560,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 2, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [14]
        TuningConfigSM90(
            M=1,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [15]
        TuningConfigSM90(
            M=2,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [16]
        TuningConfigSM90(
            M=4,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [17]
        TuningConfigSM90(
            M=8,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [18]
        TuningConfigSM90(
            M=16,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [19]
        TuningConfigSM90(
            M=32,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [20]
        TuningConfigSM90(
            M=48,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [21]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 40, mma_k),
            block_tile_shape=Index(64 * 1, 40, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [22]
        TuningConfigSM90(
            M=80,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [23]
        TuningConfigSM90(
            M=96,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [24]
        TuningConfigSM90(
            M=112,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=11,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [25]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 1, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=11,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [26]
        TuningConfigSM90(
            M=192,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 160, mma_k),
            block_tile_shape=Index(64 * 1, 160, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [27]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=13824,
            mma_shape=IndexList[3](64, 80, mma_k),
            block_tile_shape=Index(64 * 2, 80, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [28]
        TuningConfigSM90(
            M=1,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [29]
        TuningConfigSM90(
            M=2,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [30]
        TuningConfigSM90(
            M=4,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [31]
        TuningConfigSM90(
            M=8,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [32]
        TuningConfigSM90(
            M=16,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [33]
        TuningConfigSM90(
            M=32,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [34]
        TuningConfigSM90(
            M=48,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [35]
        TuningConfigSM90(
            M=64,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [36]
        TuningConfigSM90(
            M=80,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [37]
        TuningConfigSM90(
            M=96,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 1, 256, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [38]
        TuningConfigSM90(
            M=112,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [39]
        TuningConfigSM90(
            M=128,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 1, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [40]
        TuningConfigSM90(
            M=192,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 2, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [41]
        TuningConfigSM90(
            M=256,
            N=27648,
            K=5120,
            mma_shape=IndexList[3](64, 216, mma_k),
            block_tile_shape=Index(64 * 2, 216, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [42]
        TuningConfigSM90(
            M=1,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [43]
        TuningConfigSM90(
            M=2,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 1, 128, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [44]
        TuningConfigSM90(
            M=4,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [45]
        TuningConfigSM90(
            M=8,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [46]
        TuningConfigSM90(
            M=16,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [47]
        TuningConfigSM90(
            M=32,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [48]
        TuningConfigSM90(
            M=48,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [49]
        TuningConfigSM90(
            M=64,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 1, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [50]
        TuningConfigSM90(
            M=80,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [51]
        TuningConfigSM90(
            M=96,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [52]
        TuningConfigSM90(
            M=112,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [53]
        TuningConfigSM90(
            M=128,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 144, mma_k),
            block_tile_shape=Index(64 * 2, 144, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [54]
        TuningConfigSM90(
            M=192,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [55]
        TuningConfigSM90(
            M=256,
            N=76032,
            K=5120,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [56]
        TuningConfigSM90(
            M=128,
            N=1536,
            K=4096,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(64 * 1, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(1),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [57]
        TuningConfigSM90(
            M=128,
            N=4096,
            K=1536,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(64 * 2, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(132, H100.sm_count // 132),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(1),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [58]
        TuningConfigSM90(
            M=128,
            N=1536,
            K=4608,
            mma_shape=IndexList[3](64, 32, mma_k),
            block_tile_shape=Index(64 * 1, 32, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(1),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [59]
        TuningConfigSM90(
            M=64,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [60]
        TuningConfigSM90(
            M=128,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [61]
        TuningConfigSM90(
            M=256,
            N=2560,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [62]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                40 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [63]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                40 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [64]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=3584,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                80 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [65]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [66]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                40 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [67]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=27648,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                80 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [68]
        TuningConfigSM90(
            M=64,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [69]
        TuningConfigSM90(
            M=128,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                128 // size_factor,
                BK,
            ),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [70]
        TuningConfigSM90(
            M=256,
            N=13824,
            K=5120,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                256 // size_factor,
                BK,
            ),
            cluster_shape=Index(2, 2, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [71]
        TuningConfigSM90(
            M=64,
            N=3200,
            K=6400,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [72]
        TuningConfigSM90(
            M=128,
            N=3200,
            K=6400,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [73]
        TuningConfigSM90(
            M=256,
            N=3200,
            K=6400,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [74]
        TuningConfigSM90(
            M=64,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [75]
        TuningConfigSM90(
            M=128,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [76]
        TuningConfigSM90(
            M=256,
            N=6400,
            K=3200,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                128 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [77]
        TuningConfigSM90(
            M=64,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [78]
        TuningConfigSM90(
            M=128,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [79]
        TuningConfigSM90(
            M=256,
            N=3200,
            K=4992,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [80]
        TuningConfigSM90(
            M=64,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [81]
        TuningConfigSM90(
            M=128,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=9,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [82]
        TuningConfigSM90(
            M=256,
            N=3200,
            K=4608,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [83]
        TuningConfigSM90(
            M=64,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 16 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                16 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [84]
        TuningConfigSM90(
            M=128,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [85]
        TuningConfigSM90(
            M=256,
            N=1664,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [86]
        TuningConfigSM90(
            M=64,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 16 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                16 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [87]
        TuningConfigSM90(
            M=128,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 32 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                32 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=10,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [88]
        TuningConfigSM90(
            M=256,
            N=1536,
            K=3200,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [89]
        TuningConfigSM90(
            M=64,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [90]
        TuningConfigSM90(
            M=128,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 40 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                40 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [91]
        TuningConfigSM90(
            M=256,
            N=5120,
            K=75837,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                80 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [92]
        TuningConfigSM90(
            M=64,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                128 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [93]
        TuningConfigSM90(
            M=128,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 128 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                128 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [94]
        TuningConfigSM90(
            M=256,
            N=12800,
            K=2560,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                256 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=True,
            grid_shape=Index(128, H100.sm_count // 128),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(2),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [95]
        TuningConfigSM90(
            M=16,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(3),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [96]
        TuningConfigSM90(
            M=64,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 64 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 1,
                64 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=8,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(3),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [97]
        TuningConfigSM90(
            M=512,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 80 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                80 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 2, 1),
            num_pipeline_stages=8,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(3),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [98]
        TuningConfigSM90(
            M=4096,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                256 // size_factor,
                BK,
            ),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=None,
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(3),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [99]
        TuningConfigSM90(
            M=8192,
            N=2560,
            K=8192,
            mma_shape=IndexList[3](64, 256 // size_factor, mma_k),
            block_tile_shape=Index(
                64 * 2,
                256 // size_factor,
                BK,
            ),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(10, H100.sm_count // 10),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(3),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [100]
        TuningConfigSM90(
            M=16,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [101]
        TuningConfigSM90(
            M=16,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [102]
        TuningConfigSM90(
            M=16,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 1, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [103]
        TuningConfigSM90(
            M=16,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 1, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [104]
        TuningConfigSM90(
            M=32,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=1,
            partitioned_multicast=True,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [105]
        TuningConfigSM90(
            M=32,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 64, mma_k),
            block_tile_shape=Index(64 * 1, 64, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=12,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [106]
        TuningConfigSM90(
            M=32,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 112, mma_k),
            block_tile_shape=Index(64 * 1, 112, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(3),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [107]
        TuningConfigSM90(
            M=32,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 1, 96, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=1,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if True else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if True else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [108]
        TuningConfigSM90(
            M=224,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [109]
        TuningConfigSM90(
            M=224,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [110]
        TuningConfigSM90(
            M=224,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [111]
        TuningConfigSM90(
            M=224,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [112]
        TuningConfigSM90(
            M=256,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 128, mma_k),
            block_tile_shape=Index(64 * 2, 128, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [113]
        TuningConfigSM90(
            M=256,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [114]
        TuningConfigSM90(
            M=256,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=7,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [115]
        TuningConfigSM90(
            M=256,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 96, mma_k),
            block_tile_shape=Index(64 * 2, 96, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [116]
        TuningConfigSM90(
            M=288,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [117]
        TuningConfigSM90(
            M=288,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [118]
        TuningConfigSM90(
            M=288,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [119]
        TuningConfigSM90(
            M=512,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [120]
        TuningConfigSM90(
            M=512,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [121]
        TuningConfigSM90(
            M=512,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=5,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [122]
        TuningConfigSM90(
            M=512,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=6,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(2, H100.sm_count // 2),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [123]
        TuningConfigSM90(
            M=1024,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [124]
        TuningConfigSM90(
            M=1024,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [125]
        TuningConfigSM90(
            M=1024,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 168, mma_k),
            block_tile_shape=Index(64 * 2, 168, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [126]
        TuningConfigSM90(
            M=2000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [127]
        TuningConfigSM90(
            M=2000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [128]
        TuningConfigSM90(
            M=2000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [129]
        TuningConfigSM90(
            M=2000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [130]
        TuningConfigSM90(
            M=2048,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [131]
        TuningConfigSM90(
            M=2048,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [132]
        TuningConfigSM90(
            M=2048,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [133]
        TuningConfigSM90(
            M=2048,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [134]
        TuningConfigSM90(
            M=3072,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [135]
        TuningConfigSM90(
            M=3072,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [136]
        TuningConfigSM90(
            M=3072,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [137]
        TuningConfigSM90(
            M=3072,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [138]
        TuningConfigSM90(
            M=3500,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [139]
        TuningConfigSM90(
            M=3500,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [140]
        TuningConfigSM90(
            M=3500,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [141]
        TuningConfigSM90(
            M=3500,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [142]
        TuningConfigSM90(
            M=4096,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [143]
        TuningConfigSM90(
            M=4096,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [144]
        TuningConfigSM90(
            M=4096,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [145]
        TuningConfigSM90(
            M=4096,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [146]
        TuningConfigSM90(
            M=7000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [147]
        TuningConfigSM90(
            M=7000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [148]
        TuningConfigSM90(
            M=7000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(1, H100.sm_count // 1),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [149]
        TuningConfigSM90(
            M=7000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [150]
        TuningConfigSM90(
            M=8192,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [151]
        TuningConfigSM90(
            M=8192,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [152]
        TuningConfigSM90(
            M=8192,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [153]
        TuningConfigSM90(
            M=8192,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(6, H100.sm_count // 6),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [154]
        TuningConfigSM90(
            M=48000,
            N=5376,
            K=4096,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [155]
        TuningConfigSM90(
            M=48000,
            N=8192,
            K=5376,
            mma_shape=IndexList[3](64, 256, mma_k),
            block_tile_shape=Index(64 * 2, 256, BK),
            cluster_shape=Index(1, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, H100.sm_count // 4),
            schedule=MatmulSchedule(0),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [156]
        TuningConfigSM90(
            M=48000,
            N=43008,
            K=5376,
            mma_shape=IndexList[3](64, 192, mma_k),
            block_tile_shape=Index(64 * 2, 192, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(16, H100.sm_count // 16),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # Automatically generated from [max/kernels/src/linalg/matmul/gpu/sm90/tuning_table_bf16.yaml]
        # index: [157]
        TuningConfigSM90(
            M=48000,
            N=5376,
            K=21504,
            mma_shape=IndexList[3](64, 224, mma_k),
            block_tile_shape=Index(64 * 2, 224, BK),
            cluster_shape=Index(2, 1, 1),
            num_pipeline_stages=4,
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, H100.sm_count // 8),
            schedule=MatmulSchedule(2),
            splits=OptionalReg[Int](2) if False else None,
            raster_order=OptionalReg[RasterOrder](
                RasterOrder.AlongM
            ) if False else None,
            dispatch_group=TuningGroup(4),
        ),
        # ----------------END-TUNING-LIST-BF16-SMALL----------------
    ]

    return materialize[config_list]()
