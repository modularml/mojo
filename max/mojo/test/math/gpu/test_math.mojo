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

from std.math import *

from max.gpu.host import DeviceContext
from std.testing import TestSuite


def run_func[
    dtype: DType,
    kernel_fn: def[fn_dtype: DType, width: SIMDLength](
        SIMD[fn_dtype, width]
    ) thin -> SIMD[fn_dtype, width] where fn_dtype.is_floating_point(),
](
    ctx: DeviceContext, val: Scalar[dtype] = 0
) raises where dtype.is_floating_point():
    @__parameter
    def kernel(
        output: Pointer[Scalar[dtype], MutAnyOrigin], input: Scalar[dtype]
    ):
        output[unsafe_offset=0] = kernel_fn(input)

    var out = ctx.enqueue_create_buffer[dtype](1)
    ctx.enqueue_function[kernel](out, val, grid_dim=1, block_dim=1)
    ctx.synchronize()

    _ = out


def hypot_fn(val: SIMD) -> type_of(val) where val.dtype.is_floating_point():
    return hypot(val, val)


def remainder_fn(val: SIMD) -> type_of(val) where val.dtype.is_floating_point():
    return remainder(val, val)


def scalb_fn(val: SIMD) -> type_of(val) where val.dtype.is_floating_point():
    return scalb(val, val)


def gcd_fn(val: SIMD) -> type_of(val):
    return type_of(val)(gcd(Int(val), Int(val)))


def lcm_fn(val: SIMD) -> type_of(val):
    return type_of(val)(lcm(Int(val), Int(val)))


def sqrt_fn(val: SIMD) -> type_of(val):
    return sqrt(val)


def ldexp_fn(val: SIMD) -> type_of(val) where val.dtype.is_floating_point():
    return ldexp(val, 1)


def frexp_fn(val: SIMD) -> type_of(val) where val.dtype.is_floating_point():
    return frexp(val)[0]


def floor_fn(val: SIMD) -> type_of(val):
    return floor(val)


def ceil_fn(val: SIMD) -> type_of(val):
    return floor(val)


def pow_fn(val: SIMD) -> type_of(val):
    return val**val


def powi_fn(val: SIMD) -> type_of(val):
    return val**9


def powf_fn(val: SIMD) -> type_of(val):
    return val ** type_of(val)(3.2)


def test_math() raises:
    with DeviceContext() as ctx:

        @__parameter
        def test[
            *kernel_fns: def[fn_dtype: DType, width: SIMDLength](
                SIMD[fn_dtype, width]
            ) thin -> SIMD[fn_dtype, width] where fn_dtype.is_floating_point()
        ](ctx: DeviceContext) raises:
            comptime ls = kernel_fns.size

            comptime for idx in range(ls):
                comptime kernel_fn = kernel_fns[idx]
                run_func[.float32, kernel_fn[...]](ctx)
                run_func[.float16, kernel_fn[...]](ctx)

        # Anything that's commented does not work atm and needs to be
        # implemented. This list is also not exhaustive and needs to be
        # expanded.
        test[
            sqrt_fn,
            rsqrt,
            ldexp_fn,
            frexp_fn,
            log,
            log2,
            log10,
            log1p,
            # logb,
            cbrt,
            # hypot_fn,
            erfc,
            # lgamma,
            # gamma,
            # remainder_fn,
            # j0,
            # j1,
            # y0,
            # y1,
            # scalb_fn,
            gcd_fn,
            sin,
            asin,
            cos,
            acos,
            cosh,
            sinh,
            tanh,
            atanh,
            exp,
            erf,
            floor_fn,
            ceil_fn,
            pow_fn,
            powi_fn,
            powf_fn,
            recip,
        ](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
