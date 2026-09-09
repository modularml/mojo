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

from linalg.matmul.gpu.amd.warp_spec_matmul import warp_specialized_matmul
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from std.testing import assert_equal
from std.random import random_si64


def test_warp_specialization_amd[
    M: Int,
    N: Int,
    K: Int,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    WK: Int,
    a_producer_warps: Int,
    b_producer_warps: Int,
    consumer_warps: Int,
    pipeline_stages: Int = 1,
](ctx: DeviceContext) raises:
    var device_a = ctx.enqueue_create_buffer[.bfloat16](M * K)
    var device_b = ctx.enqueue_create_buffer[.bfloat16](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](M * N)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](M * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(M * K):
            var val = random_si64(0, 20)
            host_a[i] = val.cast[.bfloat16]()

        for i in range(K * N):
            var val = random_si64(0, 20)
            host_b[i] = val.cast[.bfloat16]()

    var a_tt = TileTensor(device_a, row_major[M, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[M, N]())
    var c_ref_tt = TileTensor(device_c_ref, row_major[M, N]())

    warp_specialized_matmul[
        M,
        N,
        K,
        BM,
        BN,
        BK,
        WM,
        WN,
        WK,
        a_producer_warps,
        b_producer_warps,
        consumer_warps,
        pipeline_stages,
    ](
        a_tt,
        b_tt,
        c_tt,
        ctx,
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tt,
        a_tt,
        b_tt,
        c_row_major=True,
        transpose_b=True,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        var errors = 0
        for i in range(M * N):
            # print(i // N, i % N, c_host.tensor.data[i], c_host_ref.tensor.data[i])
            if host_c[i] != host_c_ref[i]:  # and errors < 100:
                # print(i // N, i % N, host_c[i], host_c_ref[i])
                errors += 1

            # if errors < 100:
            #     print(i//N, i%N, host_c[i], host_c_ref[i])

        assert_equal(errors, 0)


def main() raises:
    with DeviceContext() as ctx:
        print("Running AMD Warp Specialization Tests")
        # test_warp_specialization_amd[
        #     4096,
        #     4096,
        #     4096,
        #     128,
        #     128,
        #     64,
        #     64,
        #     64,
        #     64,
        #     2,
        #     2,
        #     4,
        #     pipeline_stages=2,
        # ](ctx)

        test_warp_specialization_amd[
            4096, 4096, 4096, 64, 64, 64, 32, 32, 64, 2, 2, 4, pipeline_stages=2
        ](ctx)

        test_warp_specialization_amd[
            1024, 1024, 256, 64, 64, 64, 32, 32, 64, 2, 2, 4, pipeline_stages=2
        ](ctx)

        test_warp_specialization_amd[
            32, 32, 32, 32, 32, 32, 16, 16, 32, 2, 2, 2
        ](ctx)
        print("==== AMD Warp Specialization Tests passed ====")
