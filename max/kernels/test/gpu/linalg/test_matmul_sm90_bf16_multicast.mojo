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

from max.gpu.host import DeviceContext
from linalg.matmul.gpu.sm90.testbed import test_matmul_sm90
from linalg.matmul.gpu.tile_scheduler import MatmulSchedule

from std.utils.index import Index

from layout import Idx

# Helper to calculate block_tile_shape based on dtype and wgmma_n
comptime block_tile_shape[wgmma_n: Int, a_dtype: DType] = Index(
    128, wgmma_n, 128
) if a_dtype == .float8_e4m3fn else Index(128, wgmma_n, 64)

# Helper to calculate wgmma_shape based on dtype and wgmma_n
comptime wgmma_shape[wgmma_n: Int, a_dtype: DType] = Index(
    64, wgmma_n, 32
) if a_dtype == .float8_e4m3fn else Index(64, wgmma_n, 16)


def main() raises:
    with DeviceContext() as ctx:
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[80, .bfloat16],
            wgmma_shape[80, .bfloat16],
            num_consumer=2,
            num_pipeline_stages=8,
            partitioned_multicast=False,
            grid_shape=Index(32, 4),
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(512), Idx[2560], Idx[8192])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(10, 13),
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(8192), Idx[2560], Idx[8192])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(4096), Idx[2560], Idx[8192])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, 33),
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(8192), Idx[8192], Idx[2048])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(4096), Idx[8192], Idx[2048])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](ctx, Int(4096), Idx[8192], Idx[2048])

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, 16),
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Int(8192),
            Idx[14336],
            Idx[8192],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(8, 16),
            use_tma_store=True,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Int(8192),
            Idx[14336],
            Idx[8192],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Int(4096),
            Idx[14336],
            Idx[8192],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Int(4096),
            Idx[14336],
            Idx[8192],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, 33),
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[8192],
            Idx[8192],
            Idx[7168],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            grid_shape=Index(4, 33),
            use_tma_store=True,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[8192],
            Idx[8192],
            Idx[7168],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[4096],
            Idx[8192],
            Idx[7168],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=False,
            use_tma_store=True,
            schedule=MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[4096],
            Idx[8192],
            Idx[7168],
        )

        comptime for multicast_mode in range(2):
            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(1, 2, 1),
                block_tile_shape[80, .bfloat16],
                wgmma_shape[80, .bfloat16],
                num_consumer=2,
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[256],
                Idx[80],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(1, 2, 1),
                block_tile_shape[256, .bfloat16],
                wgmma_shape[256, .bfloat16],
                num_consumer=2,
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[256],
                Idx[256],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(1, 2, 1),
                block_tile_shape[64, .bfloat16],
                wgmma_shape[64, .bfloat16],
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[256],
                Idx[64],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(2, 1, 1),
                block_tile_shape[256, .bfloat16],
                wgmma_shape[256, .bfloat16],
                num_consumer=2,
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[128],
                Idx[512],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(2, 1, 1),
                block_tile_shape[64, .bfloat16],
                wgmma_shape[64, .bfloat16],
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[128],
                Idx[128],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(2, 2, 1),
                block_tile_shape[256, .bfloat16],
                wgmma_shape[256, .bfloat16],
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[256],
                Idx[512],
                Idx[128],
            )

            test_matmul_sm90[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                Index(2, 2, 1),
                block_tile_shape[64, .bfloat16],
                wgmma_shape[64, .bfloat16],
                num_consumer=2,
                partitioned_multicast=Bool(multicast_mode),
                measure_threshold=0.001,
            ](
                ctx,
                Idx[256],
                Idx[128],
                Idx[128],
            )

        print("# 2x1 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[1024],
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(1024),
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(199),
            Idx[1024],
            Idx[1024],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(200),
            Idx[512],
            Idx[256],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 2, 1),
            block_tile_shape[64, .bfloat16],
            wgmma_shape[64, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(201),
            Idx[2048],
            Idx[256],
        )

        print("# 1x2 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[1024],
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(1024),
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(99),
            Idx[1024],
            Idx[1024],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(100),
            Idx[512],
            Idx[256],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, .bfloat16],
            wgmma_shape[128, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(201),
            Idx[2048],
            Idx[256],
        )

        print("# 2x2 warp specialized gemm with multicasting tests")

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Idx[1024],
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(1024),
            Idx[512],
            Idx[128],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(199),
            Idx[1024],
            Idx[1024],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(200),
            Idx[512],
            Idx[256],
        )

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(2, 2, 1),
            block_tile_shape[256, .bfloat16],
            wgmma_shape[256, .bfloat16],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            Int(201),
            Idx[2048],
            Idx[256],
        )
