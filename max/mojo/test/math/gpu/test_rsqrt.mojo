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

from std.math import rsqrt, sqrt
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu import *
from max.gpu.host import DeviceContext, HostBuffer, get_gpu_target
from std.testing import *

from std.utils import IndexList
from std.utils.coord import Coord


def run_elementwise[
    dtype: DType,
    kernel_fn: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width]
    ) thin raises -> SIMD[dtype, width],
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime length = 256

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

    var in_host: HostBuffer[dtype]
    with in_device.map_to_host() as in_host2:
        for i in range(length):
            in_host2[i] = 0.001 * abs(Scalar[dtype](i) - length // 2)
        in_host = in_host2^

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx: Coord):
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx[0].value(),
            rsqrt(
                in_buffer.unsafe_ptr().unsafe_load[width=simd_width](
                    idx[0].value()
                )
            ),
        )

    elementwise[func, pack_size, target="gpu"](Coord(length), ctx)

    with out_device.map_to_host() as out_host:
        for i in range(length):
            var msg = String(
                "values did not match at position ", i, " for dtype=", dtype
            )

            comptime if dtype == .float32:
                assert_almost_equal(
                    out_host[i],
                    rsqrt(in_host[i]),
                    msg=msg,
                    atol=1e-08,
                    rtol=1e-05,
                )
            else:
                assert_almost_equal(
                    out_host[i],
                    rsqrt(in_host[i]),
                    msg=msg,
                    atol=1e-04,
                    rtol=1e-03,
                )


def test_rsqrt() raises:
    with DeviceContext() as ctx:
        run_elementwise[.float16, sqrt](ctx)
        run_elementwise[.float32, sqrt](ctx)
        run_elementwise[.float64, sqrt](ctx)
        run_elementwise[.float16, rsqrt](ctx)
        run_elementwise[.float32, rsqrt](ctx)
        run_elementwise[.float64, rsqrt](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
