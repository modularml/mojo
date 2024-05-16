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
# RUN: %mojo %s

from sys.info import has_neon
from testing import assert_equal, assert_true, assert_false, assert_almost_equal
from utils.numerics import (
    FPUtils,
    inf,
    isfinite,
    isinf,
    isnan,
    max_finite,
    nan,
    ulp,
    neg_inf,
)


# TODO: improve coverage and organization of these tests
def test_FPUtils():
    assert_equal(FPUtils[DType.float32].mantissa_width(), 23)
    assert_equal(FPUtils[DType.float32].exponent_bias(), 127)

    alias FPU64 = FPUtils[DType.float64]

    assert_equal(FPU64.mantissa_width(), 52)
    assert_equal(FPU64.exponent_bias(), 1023)

    assert_equal(FPU64.get_exponent(FPU64.set_exponent(1, 2)), 2)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(1, 3)), 3)
    assert_equal(FPU64.get_exponent(FPU64.set_exponent(-1, 4)), 4)
    assert_equal(FPU64.get_mantissa(FPU64.set_mantissa(-1, 5)), 5)
    assert_true(FPU64.get_sign(FPU64.set_sign(0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(0, False)))
    assert_true(FPU64.get_sign(FPU64.set_sign(-0, True)))
    assert_false(FPU64.get_sign(FPU64.set_sign(-0, False)))
    assert_false(FPU64.get_sign(1))
    assert_true(FPU64.get_sign(-1))
    assert_false(FPU64.get_sign(FPU64.pack(False, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(False, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(False, 6, 12)), 12)
    assert_true(FPU64.get_sign(FPU64.pack(True, 6, 12)))
    assert_equal(FPU64.get_exponent(FPU64.pack(True, 6, 12)), 6)
    assert_equal(FPU64.get_mantissa(FPU64.pack(True, 6, 12)), 12)


def test_isfinite():
    assert_true(isfinite(Float32(33)))

    @parameter
    if not has_neon():
        assert_false(isfinite(inf[DType.bfloat16]()))
        assert_false(isfinite(neg_inf[DType.bfloat16]()))
        assert_false(isfinite(nan[DType.bfloat16]()))

    assert_false(isfinite(inf[DType.float16]()))
    assert_false(isfinite(inf[DType.float32]()))
    assert_false(isfinite(inf[DType.float64]()))
    assert_false(isfinite(neg_inf[DType.float16]()))
    assert_false(isfinite(neg_inf[DType.float32]()))
    assert_false(isfinite(neg_inf[DType.float64]()))
    assert_false(isfinite(nan[DType.float16]()))
    assert_false(isfinite(nan[DType.float32]()))
    assert_false(isfinite(nan[DType.float64]()))


def test_isinf():
    assert_false(isinf(Float32(33)))

    @parameter
    if not has_neon():
        assert_true(isinf(inf[DType.bfloat16]()))
        assert_true(isinf(neg_inf[DType.bfloat16]()))
        assert_false(isinf(nan[DType.bfloat16]()))

    assert_true(isinf(inf[DType.float16]()))
    assert_true(isinf(inf[DType.float32]()))
    assert_true(isinf(inf[DType.float64]()))
    assert_true(isinf(neg_inf[DType.float16]()))
    assert_true(isinf(neg_inf[DType.float32]()))
    assert_true(isinf(neg_inf[DType.float64]()))
    assert_false(isinf(nan[DType.float16]()))
    assert_false(isinf(nan[DType.float32]()))
    assert_false(isinf(nan[DType.float64]()))


def test_isnan():
    assert_false(isnan(Float32(33)))

    @parameter
    if not has_neon():
        assert_false(isnan(inf[DType.bfloat16]()))
        assert_false(isnan(neg_inf[DType.bfloat16]()))
        assert_true(isnan(nan[DType.bfloat16]()))

    assert_false(isnan(inf[DType.float16]()))
    assert_false(isnan(inf[DType.float32]()))
    assert_false(isnan(inf[DType.float64]()))
    assert_false(isnan(neg_inf[DType.float16]()))
    assert_false(isnan(neg_inf[DType.float32]()))
    assert_false(isnan(neg_inf[DType.float64]()))
    assert_true(isnan(nan[DType.float16]()))
    assert_true(isnan(nan[DType.float32]()))
    assert_true(isnan(nan[DType.float64]()))


def test_ulp():
    assert_true(isnan(ulp(nan[DType.float32]())))
    assert_true(isinf(ulp(inf[DType.float32]())))
    assert_true(isinf(ulp(-inf[DType.float32]())))
    assert_almost_equal(ulp(Float64(0)), 5e-324)
    assert_equal(ulp(max_finite[DType.float64]()), 1.99584030953472e292)
    assert_equal(ulp(Float64(5)), 8.881784197001252e-16)
    assert_equal(ulp(Float64(-5)), 8.881784197001252e-16)


def main():
    test_FPUtils()
    test_isfinite()
    test_isinf()
    test_isnan()
    # TODO: test nextafter
    test_ulp()
