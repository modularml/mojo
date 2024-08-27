# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# REQUIRES: linux
# RUN: %mojo %s


from collections import InlineArray
from math import erf
from random import randn, seed

from test_utils import compare, libm_call
from testing import assert_almost_equal, assert_equal


def test_erf_float32():
    assert_equal(erf(Float32(0)), 0.0)
    assert_almost_equal(erf(SIMD[DType.float32, 2](2)), 0.995322)
    assert_almost_equal(erf(Float32(0.1)), 0.112462)
    assert_almost_equal(erf(Float32(-0.1)), -0.112462)
    assert_almost_equal(erf(Float32(-1)), -0.8427007)
    assert_almost_equal(erf(Float32(-2)), -0.995322)


def test_erf_float64():
    assert_equal(erf(Float64(0)), 0.0)
    assert_almost_equal(erf(SIMD[DType.float64, 2](2)), 0.995322)
    assert_almost_equal(erf(Float64(0.1)), 0.112462)
    assert_almost_equal(erf(Float64(-0.1)), -0.112462)
    assert_almost_equal(erf(Float64(-1)), -0.8427007)
    assert_almost_equal(erf(Float64(-2)), -0.995322)


def test_erf_libm():
    seed(0)
    var N = 8192
    alias test_dtype = DType.float32

    # generate input values and write them to file
    var x32 = UnsafePointer[Scalar[test_dtype]].alloc(N)
    randn[test_dtype](x32, N, 0, 9.0)
    print("For N=" + str(N) + " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # math.erf result
    ####################
    var y32 = UnsafePointer[Scalar[test_dtype]].alloc(N)
    for i in range(N):
        y32[i] = erf(x32[i])  # math.erf

    ####################
    ## libm erf result
    ####################
    @always_inline
    fn erf_libm[
        type: DType, simd_width: Int
    ](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return libm_call[type, simd_width, "erff", "err"](arg)

    var libm_out = UnsafePointer[Scalar[test_dtype]].alloc(N)
    for i in range(N):
        libm_out[i] = erf_libm(x32[i])

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[test_dtype, 4](
        0.0, 5.9604644775390625e-08, 0.0, 1.172195140952681e-07
    )

    var err = compare[test_dtype](
        y32, libm_out, N, msg="Compare Mojo math.erf vs. LibM"
    )

    assert_almost_equal(err, abs_rel_err)

    x32.free()
    y32.free()
    libm_out.free()


def main():
    test_erf_float32()
    test_erf_float64()
    test_erf_libm()
