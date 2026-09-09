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
"""2SM (cta_group=2) tests for block_scaled_matmul_small_bn.

Tests the 2CTA cooperative MMA path where two SMs work on a single MMA
instruction. Each CTA loads its own BN=MMA_N/2 columns of B data but
the full MMA_N scale factors.
"""

from std.sys import size_of
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Idx
from linalg.matmul.gpu.sm100.testbed_block_scaled_fp4 import (
    test_blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE


def run_matmul_sm100_block_scaled_fp4_small_bn_2sm_suite[
    suite_scales_dtype: DType,
    suite_sf_vector_size: Int,
]() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = suite_scales_dtype
        comptime SF_VECTOR_SIZE = suite_sf_vector_size
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        # 2SM tests: sweep MMA_N in [16, 32].
        # Note: MMA_N=24 is valid HW but BN=12 breaks TMA tile layout.
        comptime for mma_n in [16, 32, 48, 64, 96]:
            comptime block_tile = Index(128, mma_n // 2, BK)
            comptime umma = Index(256, mma_n, MMA_K)

            # Basic cluster_shape=(2,1,1)
            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[2304], Idx[16384])

            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[16384], Idx[2048])

            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[6656], Idx[16384])

            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[16384], Idx[6656])

            # Larger cluster shapes
            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(4), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=4,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[2304], Idx[16384])

            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), Int32(2), 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=4,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
            ](ctx, Idx[1], Idx[6656], Idx[16384])

        # Kimi-K2.5 shape that triggered race condition
        comptime kimi_block_tile = Index(128, 8, BK)
        comptime kimi_umma = Index(256, 16, MMA_K)
        test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            dtype,
            dtype,
            out_dtype,
            scales_dtype,
            kimi_block_tile,
            kimi_umma,
            cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
            cta_group=2,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            swapAB=True,
            k_group_size=1,
            num_clc_pipeline_stages=0,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            is_small_bn=True,
        ](ctx, Idx[1], Idx[36864], Idx[128 * 28])

        # Epilogue fusion tests: verify TileWriter's elementwise_lambda_fn path
        # with 2SM (cta_group=2).
        print("\n--- 2SM Epilogue fusion tests ---")
        comptime for mma_n in [16, 32, 48, 64, 96]:
            comptime epi_block_tile = Index(128, mma_n // 2, BK)
            comptime epi_umma = Index(256, mma_n, MMA_K)

            test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                epi_block_tile,
                epi_umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                swapAB=True,
                k_group_size=2,
                num_clc_pipeline_stages=0,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                is_small_bn=True,
                normal_epilogue=True,
            ](ctx, Idx[1], Idx[2304], Idx[16384])


def main() raises:
    run_matmul_sm100_block_scaled_fp4_small_bn_2sm_suite[
        suite_scales_dtype=NVFP4_SF_DTYPE,
        suite_sf_vector_size=NVFP4_SF_VECTOR_SIZE,
    ]()
