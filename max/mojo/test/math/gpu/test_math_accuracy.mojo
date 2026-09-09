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

from std.math import cosh, exp, exp2, log, sinh
from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu import *
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.testing import assert_almost_equal, assert_equal, TestSuite

from std.utils import Index, IndexList
from std.utils.coord import Coord

comptime length = 8192


def run_elementwise[
    dtype: DType,
    math_fn: def[fn_dtype: DType, fn_width: SIMDLength](
        SIMD[fn_dtype, fn_width]
    ) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point(),
](
    ctx: DeviceContext, in_device: DeviceBuffer[dtype]
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var out_device = ctx.enqueue_create_buffer[dtype](length)

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = Int(idx0[0].value())
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var result = math_fn(val)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, result)

    elementwise[func, pack_size, target="gpu"](Coord(length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = math_fn(in_host[i])

            comptime atol = 1e-05 if dtype == DType.float32 else 1e-4
            comptime rtol = 2e-05 if dtype == DType.float32 else 2e-2
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=atol,
                rtol=rtol,
            )


def _test_exp[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var input = ctx.enqueue_create_buffer[dtype](length)
    comptime epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[dtype](i) + epsilon)
    run_elementwise[dtype, exp](ctx, input)


def _test_exp2[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var input = ctx.enqueue_create_buffer[dtype](length)
    comptime epsilon = 0.001
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = log(Scalar[dtype](i) + epsilon)
    run_elementwise[dtype, exp2](ctx, input)


def _test_cosh[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var input = ctx.enqueue_create_buffer[dtype](length)
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = (Scalar[dtype](i) / length * 10) - 5
    run_elementwise[dtype, cosh](ctx, input)


def _test_sinh[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var input = ctx.enqueue_create_buffer[dtype](length)
    with input.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = (Scalar[dtype](i) / length * 10) - 5
    run_elementwise[dtype, sinh](ctx, input)


def test_math_accuracy() raises:
    with DeviceContext() as ctx:
        _test_exp[.float32](ctx)
        _test_exp[.float16](ctx)
        _test_exp[.bfloat16](ctx)
        _test_exp2[.float32](ctx)
        _test_exp2[.float16](ctx)
        _test_exp2[.bfloat16](ctx)
        _test_cosh[.float32](ctx)
        _test_cosh[.float16](ctx)
        _test_cosh[.bfloat16](ctx)
        _test_sinh[.float32](ctx)
        _test_sinh[.float16](ctx)
        _test_sinh[.bfloat16](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
