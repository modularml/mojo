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

from std.math import erf
from std.random import randn, seed

from test_utils import compare, libm_call
from std.testing import assert_almost_equal, assert_equal, TestSuite


def test_erf_float32() raises:
    assert_equal(erf(Float32(0)), 0.0)
    assert_almost_equal(erf(SIMD[.float32, 2](2)), 0.995322)
    assert_almost_equal(erf(Float32(0.1)), 0.112462)
    assert_almost_equal(erf(Float32(-0.1)), -0.112462)
    assert_almost_equal(erf(Float32(-1)), -0.8427007)
    assert_almost_equal(erf(Float32(-2)), -0.995322)


def test_erf_float64() raises:
    assert_equal(erf(Float64(0)), 0.0)
    assert_almost_equal(erf(SIMD[.float64, 2](2)), 0.995322)
    assert_almost_equal(erf(Float64(0.1)), 0.112462)
    assert_almost_equal(erf(Float64(-0.1)), -0.112462)
    assert_almost_equal(erf(Float64(-1)), -0.8427007)
    assert_almost_equal(erf(Float64(-2)), -0.995322)


def test_erf_libm() raises:
    seed(0)
    var N = 8192
    comptime test_dtype = DType.float32

    # generate input values and write them to file
    var x32_allocation = alloc[Scalar[test_dtype]]({count = N}).into_managed()
    var x32 = x32_allocation.unsafe_ptr()
    randn[test_dtype](x32, N, 0, 9.0)
    print("For N=", N, " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # math.erf result
    ####################
    var y32_allocation = alloc[Scalar[test_dtype]]({count = N}).into_managed()
    var y32 = y32_allocation.unsafe_ptr()
    for i in range(N):
        y32[unsafe_offset=i] = erf(x32[unsafe_offset=i])  # math.erf

    ####################
    ## libm erf result
    ####################
    @always_inline
    def erf_libm[
        dtype: DType, simd_width: SIMDLength
    ](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return libm_call["erff", "err"](arg)

    var libm_out_allocation = alloc[Scalar[test_dtype]](
        {count = N}
    ).into_managed()
    var libm_out = libm_out_allocation.unsafe_ptr()
    for i in range(N):
        libm_out[unsafe_offset=i] = erf_libm(x32[unsafe_offset=i])

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[test_dtype, 4](
        0.0, 5.9604644775390625e-08, 0.0, 1.172195140952681e-07
    )

    var err = compare[test_dtype](
        y32, libm_out, N, msg="Compare Mojo math.erf vs. LibM"
    )

    assert_almost_equal(err, abs_rel_err)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
