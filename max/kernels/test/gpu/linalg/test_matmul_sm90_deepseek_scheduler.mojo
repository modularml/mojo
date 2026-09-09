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

# Helper to calculate wgmma_shape based on dtype and BN
comptime wgmma_shape[BN: Int, a_dtype: DType] = Index(
    64, BN, 32
) if a_dtype == .float8_e4m3fn else Index(64, BN, 16)

# Helper to calculate num_consumer based on BM
comptime get_num_consumer[BM: Int] = 1 if BM == 64 else 2


def main() raises:
    with DeviceContext() as ctx:
        # NOTE: please note that cublaslt handle should be used for fp8-e4m3fn and cublas handle for bfloat16
        # because cublas does not support float8-e4m3fn. Also, fp8 tests should be run first and then bfloat16 tests
        # otherwise we will get unhandled exception error.
        print("FP8-E4M3FN GEMM TESTS")
        test_matmul_sm90[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 80, 128),
            wgmma_shape[80, .float8_e4m3fn],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=6,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](
            ctx,
            Idx[512],
            Idx[2560],
            Idx[8192],
        )
        test_matmul_sm90[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 128, 128),
            wgmma_shape[128, .float8_e4m3fn],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=8,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](
            ctx,
            Idx[512],
            Idx[16384],
            Idx[2048],
        )
        test_matmul_sm90[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 48, 128),
            wgmma_shape[48, .float8_e4m3fn],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=8,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](
            ctx,
            Idx[512],
            Idx[2304],
            Idx[16384],
        )
        test_matmul_sm90[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 128, 128),
            wgmma_shape[128, .float8_e4m3fn],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=4,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](
            ctx,
            Idx[512],
            Idx[13312],
            Idx[16384],
        )
        test_matmul_sm90[
            .float8_e4m3fn,
            .float8_e4m3fn,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 128, 128),
            wgmma_shape[128, .float8_e4m3fn],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=4,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](
            ctx,
            Idx[512],
            Idx[16384],
            Idx[6656],
        )

        print("BFLOAT16 GEMM TESTS")
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 32, 64),
            wgmma_shape[32, .bfloat16],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=8,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(64), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 40, 64),
            wgmma_shape[40, .bfloat16],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=8,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(128), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 80, 64),
            wgmma_shape[80, .bfloat16],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=8,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(256), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 160, 64),
            wgmma_shape[160, .bfloat16],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=7,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(512), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(64, 160, 64),
            wgmma_shape[160, .bfloat16],
            num_consumer=get_num_consumer[64],
            num_pipeline_stages=5,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(1024), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 160, 64),
            wgmma_shape[160, .bfloat16],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=5,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(2048), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 256, 64),
            wgmma_shape[256, .bfloat16],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=4,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(4096), Idx[2560], Idx[8192])
        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            Index(128, 256, 64),
            wgmma_shape[256, .bfloat16],
            num_consumer=get_num_consumer[128],
            num_pipeline_stages=4,
            partitioned_multicast=False,
            schedule=MatmulSchedule.DS_SCHEDULER,
            grid_shape=Index(128, 1),
            measure_threshold=0.001,
        ](ctx, Int(8192), Idx[2560], Idx[8192])
