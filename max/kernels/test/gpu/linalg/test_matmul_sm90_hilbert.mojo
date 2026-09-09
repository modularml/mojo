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

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from linalg.matmul.gpu.sm90.testbed import test_matmul_sm90

from std.utils.index import Index

from layout import Idx

# NOTE: This test originally tested hilbert_swizzle=True functionality,
# but the testbed doesn't currently support the hilbert_swizzle parameter.
# To properly test hilbert swizzle, the testbed would need to be updated
# to include this parameter and pass it to warp_specialize_gemm_with_multicasting.

# Helper to calculate block_tile_shape - fixed for bfloat16
comptime block_tile_shape[wgmma_n: Int] = Index(128, wgmma_n, 64)

# Helper to calculate wgmma_shape - fixed for bfloat16
comptime wgmma_shape[wgmma_n: Int] = Index(64, wgmma_n, 16)


def main() raises:
    with DeviceContext() as ctx:
        comptime M = 8192
        comptime N = 6144
        comptime K = 4096

        print(
            "Running warp specialize gemm test (Note: hilbert swizzle"
            " not supported in testbed)"
        )
        print("Test configuration: M=", M, ", N=", N, ", K=", K)

        test_matmul_sm90[
            .bfloat16,
            .bfloat16,
            .bfloat16,
            Index(1, 1, 1),
            block_tile_shape[64],
            wgmma_shape[64],
        ](ctx, Int(M), Idx[N], Idx[K])

        print("Test completed successfully")
