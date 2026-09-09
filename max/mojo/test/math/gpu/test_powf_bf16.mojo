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

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu import *
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import assert_almost_equal, TestSuite

from std.utils import IndexList
from std.utils.coord import Coord

comptime type = DType.float32


def run_elementwise(exponent: BFloat16, ctx: DeviceContext) raises:
    comptime length = 256

    comptime pack_size = simd_width_of[type, target=get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[type](length)
    var out_device = ctx.enqueue_create_buffer[type](length)

    # Add a small constant to avoid 0^-pow.
    comptime epsilon = 0.001
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = abs((Scalar[type](i) - length // 2) + epsilon)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(out_buffer, in_buffer, exponent)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = Int(idx0[0].value())

        var val = (
            in_buffer.unsafe_ptr()
            .unsafe_load[width=simd_width](idx)
            .cast[.bfloat16]()
        )
        var result = val ** SIMD[.bfloat16, simd_width](exponent)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, result.cast[.float32]()
        )

    elementwise[func, pack_size, target="gpu"](Coord(length), ctx)
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = in_host[i] ** exponent.cast[.float32]()
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=1e-04,
                rtol=2e-02,
            )


def test_powf_bf16() raises:
    # NOTE: This is expected to fail. Keeping this around as a negative test
    # so we know when its fixed.
    with DeviceContext() as ctx:
        run_elementwise(0.375, ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
