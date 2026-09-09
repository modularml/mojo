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
from std.math.fast import exp_approx_f32
from std.math import exp as ref_exp
from max.algorithm.functional import elementwise
from max.gpu import *
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import *
from std.utils import Index, IndexList
from std.utils.coord import Coord


@__parameter
def run_exp_approx_test[
    simd_width: Int
](ctx: DeviceContext, *, half_range: Float32, rtol: Float64) raises:
    comptime dtype = DType.float32
    comptime length = 256

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

    # Fill test data with a sweep symmetric around zero spanning
    # [-half_range, half_range]. See `main` for the ranges exercised.
    var step = half_range / Scalar[dtype](length // 2)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = step * (Scalar[dtype](i) - length // 2)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = Int(idx0[0].value())
        var v = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, exp_approx_f32[simd_width](v)
        )

    # Launch elementwise kernel on GPU (width is compile-time parameter)
    elementwise[func, simd_width, target="gpu"](Coord(length), ctx)

    # Validate results
    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var msg = String(
                "Mismatch at index ",
                i,
                " for SIMD width=",
                simd_width,
                " value=",
                in_host[i],
            )
            assert_almost_equal(
                out_host[i],
                Scalar[dtype](ref_exp(Scalar[dtype](in_host[i]))),
                msg=msg,
                atol=1e-07,
                rtol=rtol,
            )


def main() raises:
    with DeviceContext() as ctx:
        # Full-domain sweep spanning roughly [-87, 87] (staying below the
        # float32 exp overflow threshold ~88.7).
        run_exp_approx_test[1](ctx, half_range=87.0, rtol=1.2e-02)
        run_exp_approx_test[2](ctx, half_range=87.0, rtol=1.2e-02)

        # Near-zero band, where the cubic is most accurate.
        run_exp_approx_test[1](ctx, half_range=0.128, rtol=1.5e-03)
        run_exp_approx_test[2](ctx, half_range=0.128, rtol=1.5e-03)
