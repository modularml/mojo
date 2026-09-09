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


def run_matmul_sm100_block_scaled_fp4_2sm_suite[
    suite_scales_dtype: DType,
    suite_sf_vector_size: Int,
]() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8  # TODO: (KERN-2238): Replace with float4-e2m1fn
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = suite_scales_dtype
        comptime SF_VECTOR_SIZE = suite_sf_vector_size
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B

        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        comptime for bm in [128]:
            comptime for bn in [64, 128]:
                comptime block_tile_shape = Index(bm, bn, BK)
                comptime umma_shape = Index(2 * bm, 2 * bn, MMA_K)

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1000),
                    Idx[1024],
                    Idx[1024 + 32],
                    alpha=0.5,
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=4,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(512),
                    Idx[4096],
                    Idx[1024 + 32],
                    alpha=0.135,
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=0,
                    k_group_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(500),
                    Idx[2048],
                    Idx[4096],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=2,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(999),
                    Idx[256],
                    Idx[128],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(777),
                    Idx[2560],
                    Idx[8192],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=1,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(1),
                    Idx[576],
                    Idx[7168],
                )

                # swapAB tests
                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(16),
                    Idx[1024],
                    Idx[1024 + 32],
                )

                test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                    dtype,
                    dtype,
                    out_dtype,
                    scales_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    swapAB=True,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](
                    ctx,
                    Int(100),
                    Idx[2560],
                    Idx[8192],
                )


def main() raises:
    run_matmul_sm100_block_scaled_fp4_2sm_suite[
        suite_scales_dtype=NVFP4_SF_DTYPE,
        suite_sf_vector_size=NVFP4_SF_VECTOR_SIZE,
    ]()
