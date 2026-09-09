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

from std.math import log, log2, log10
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu import *
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import assert_almost_equal, TestSuite

from std.utils import IndexList
from std.utils.coord import Coord


def run_elementwise[
    dtype: DType,
    log_fn: def[fn_dtype: DType, fn_width: SIMDLength](
        SIMD[fn_dtype, fn_width]
    ) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point(),
](ctx: DeviceContext) raises where dtype.is_floating_point():
    comptime length = 8192

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

    comptime epsilon = 0.001
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Scalar[dtype](i) + epsilon

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = Int(idx0[0].value())
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var result = log_fn(val)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, result)

    elementwise[func, pack_size, target="gpu"](Coord(length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = log_fn(in_host[i])

            comptime atol = 1e-07 if dtype == DType.float32 else 1e-4
            comptime rtol = 2e-07 if dtype == DType.float32 else 2e-2
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=atol,
                rtol=rtol,
            )


def test_log() raises:
    with DeviceContext() as ctx:
        run_elementwise[.float32, log](ctx)
        run_elementwise[.float32, log10](ctx)
        run_elementwise[.float32, log2](ctx)
        run_elementwise[.float16, log](ctx)
        run_elementwise[.float16, log10](ctx)
        run_elementwise[.float16, log2](ctx)
        run_elementwise[.bfloat16, log](ctx)
        run_elementwise[.bfloat16, log10](ctx)
        run_elementwise[.bfloat16, log2](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
