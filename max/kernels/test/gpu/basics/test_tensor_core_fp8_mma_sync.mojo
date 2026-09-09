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

from std.io.io import _printf

from max.gpu.host import DeviceContext
from max.gpu import thread_idx
from max.gpu.compute.mma import mma


def mma_sync_16x8x32_E4M3():
    var a = SIMD[.float8_e4m3fn, 16](1.0)
    var b = SIMD[.float8_e4m3fn, 8](2.0)
    var c = SIMD[.float32, 4](0.0)
    var d = SIMD[.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[.float64](),
        d[1].cast[.float64](),
        d[2].cast[.float64](),
        d[3].cast[.float64](),
    )


def test_mma_sync_16x8x32_E4M3(ctx: DeviceContext) raises:
    print("== test_mma_sync_16x8x32_E4M3")
    comptime kernel = mma_sync_16x8x32_E4M3
    ctx.enqueue_function[kernel](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


def mma_sync_16x8x32_E4M2():
    var a = SIMD[.float8_e5m2, 16](2.0)
    var b = SIMD[.float8_e5m2, 8](3.0)
    var c = SIMD[.float32, 4](0.0)
    var d = SIMD[.float32, 4](0.0)
    mma(d, a, b, c)

    _printf["thread %d : %g %g %g %g\n"](
        thread_idx.x,
        d[0].cast[.float64](),
        d[1].cast[.float64](),
        d[2].cast[.float64](),
        d[3].cast[.float64](),
    )


def test_mma_sync_16x8x32_E5M2(ctx: DeviceContext) raises:
    print("== test_mma_sync_16x8x32_E5M2")
    comptime kernel = mma_sync_16x8x32_E4M2
    ctx.enqueue_function[kernel](
        grid_dim=(1),
        block_dim=(32),
    )
    ctx.synchronize()


def main() raises:
    with DeviceContext() as ctx:
        # CHECK-LABEL: test_mma_sync_16x8x32_E4M3
        # CHECK-DAG: thread 0 : 64 64 64 64
        # CHECK-DAG: thread 1 : 64 64 64 64
        # CHECK-DAG: thread 2 : 64 64 64 64
        # CHECK-DAG: thread 3 : 64 64 64 64
        # CHECK-DAG: thread 4 : 64 64 64 64
        # CHECK-DAG: thread 5 : 64 64 64 64
        # CHECK-DAG: thread 6 : 64 64 64 64
        # CHECK-DAG: thread 7 : 64 64 64 64
        # CHECK-DAG: thread 8 : 64 64 64 64
        # CHECK-DAG: thread 9 : 64 64 64 64
        # CHECK-DAG: thread 10 : 64 64 64 64
        # CHECK-DAG: thread 11 : 64 64 64 64
        # CHECK-DAG: thread 12 : 64 64 64 64
        # CHECK-DAG: thread 13 : 64 64 64 64
        # CHECK-DAG: thread 14 : 64 64 64 64
        # CHECK-DAG: thread 15 : 64 64 64 64
        # CHECK-DAG: thread 16 : 64 64 64 64
        # CHECK-DAG: thread 17 : 64 64 64 64
        # CHECK-DAG: thread 18 : 64 64 64 64
        # CHECK-DAG: thread 19 : 64 64 64 64
        # CHECK-DAG: thread 20 : 64 64 64 64
        # CHECK-DAG: thread 21 : 64 64 64 64
        # CHECK-DAG: thread 22 : 64 64 64 64
        # CHECK-DAG: thread 23 : 64 64 64 64
        # CHECK-DAG: thread 24 : 64 64 64 64
        # CHECK-DAG: thread 25 : 64 64 64 64
        # CHECK-DAG: thread 26 : 64 64 64 64
        # CHECK-DAG: thread 27 : 64 64 64 64
        # CHECK-DAG: thread 28 : 64 64 64 64
        # CHECK-DAG: thread 29 : 64 64 64 64
        # CHECK-DAG: thread 30 : 64 64 64 64
        # CHECK-DAG: thread 31 : 64 64 64 64
        test_mma_sync_16x8x32_E4M3(ctx)
        # CHECK-LABEL: test_mma_sync_16x8x32_E5M2
        # CHECK-DAG: thread 0 : 192 192 192 192
        # CHECK-DAG: thread 1 : 192 192 192 192
        # CHECK-DAG: thread 2 : 192 192 192 192
        # CHECK-DAG: thread 3 : 192 192 192 192
        # CHECK-DAG: thread 4 : 192 192 192 192
        # CHECK-DAG: thread 5 : 192 192 192 192
        # CHECK-DAG: thread 6 : 192 192 192 192
        # CHECK-DAG: thread 7 : 192 192 192 192
        # CHECK-DAG: thread 8 : 192 192 192 192
        # CHECK-DAG: thread 9 : 192 192 192 192
        # CHECK-DAG: thread 10 : 192 192 192 192
        # CHECK-DAG: thread 11 : 192 192 192 192
        # CHECK-DAG: thread 12 : 192 192 192 192
        # CHECK-DAG: thread 13 : 192 192 192 192
        # CHECK-DAG: thread 14 : 192 192 192 192
        # CHECK-DAG: thread 15 : 192 192 192 192
        # CHECK-DAG: thread 16 : 192 192 192 192
        # CHECK-DAG: thread 17 : 192 192 192 192
        # CHECK-DAG: thread 18 : 192 192 192 192
        # CHECK-DAG: thread 19 : 192 192 192 192
        # CHECK-DAG: thread 20 : 192 192 192 192
        # CHECK-DAG: thread 21 : 192 192 192 192
        # CHECK-DAG: thread 22 : 192 192 192 192
        # CHECK-DAG: thread 23 : 192 192 192 192
        # CHECK-DAG: thread 24 : 192 192 192 192
        # CHECK-DAG: thread 25 : 192 192 192 192
        # CHECK-DAG: thread 26 : 192 192 192 192
        # CHECK-DAG: thread 27 : 192 192 192 192
        # CHECK-DAG: thread 28 : 192 192 192 192
        # CHECK-DAG: thread 29 : 192 192 192 192
        # CHECK-DAG: thread 30 : 192 192 192 192
        # CHECK-DAG: thread 31 : 192 192 192 192
        test_mma_sync_16x8x32_E5M2(ctx)
