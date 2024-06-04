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

from testing import assert_almost_equal, assert_equal, assert_false, assert_true

from utils.numerics import (
    FPUtils,
    get_accum_type,
    inf,
    isfinite,
    isinf,
    isnan,
    max_finite,
    max_or_inf,
    min_finite,
    min_or_neg_inf,
    nan,
    neg_inf,
    nextafter,
    ulp,
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


def test_get_accum_type():
    assert_equal(get_accum_type[DType.float32](), DType.float32)
    assert_equal(get_accum_type[DType.float64](), DType.float64)
    assert_equal(get_accum_type[DType.bfloat16](), DType.float32)
    assert_equal(get_accum_type[DType.int8](), DType.int8)
    assert_equal(get_accum_type[DType.int16](), DType.int16)
    assert_equal(get_accum_type[DType.int32](), DType.int32)
    assert_equal(get_accum_type[DType.int64](), DType.int64)
    assert_equal(get_accum_type[DType.uint8](), DType.uint8)
    assert_equal(get_accum_type[DType.uint16](), DType.uint16)
    assert_equal(get_accum_type[DType.uint32](), DType.uint32)
    assert_equal(get_accum_type[DType.uint64](), DType.uint64)


def test_isfinite():
    assert_true(isfinite(Float32(33)))

    # TODO(KERN-228): support BF16 on neon systems.
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

    # TODO(KERN-228): support BF16 on neon systems.
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

    # TODO(KERN-228): support BF16 on neon systems.
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


fn overflow_int[type: DType]() -> Bool:
    constrained[type.is_integral(), "comparison only valid on integral types"]()
    return max_finite[type]() + 1 < max_finite[type]()


fn overflow_fp[type: DType]() -> Bool:
    constrained[
        type.is_floating_point(),
        "comparison only valid on floating point types",
    ]()
    return max_finite[type]() + 1 == max_finite[type]()


def test_max_finite():
    assert_almost_equal(max_finite[DType.float32](), 3.4028235e38)
    assert_almost_equal(max_finite[DType.float64](), 1.7976931348623157e308)

    assert_true(overflow_int[DType.int8]())
    assert_true(overflow_int[DType.uint8]())
    assert_true(overflow_int[DType.int16]())
    assert_true(overflow_int[DType.uint16]())
    assert_true(overflow_int[DType.int32]())
    assert_true(overflow_int[DType.uint32]())
    assert_true(overflow_int[DType.int64]())
    assert_true(overflow_int[DType.uint64]())

    assert_true(overflow_fp[DType.float32]())
    assert_true(overflow_fp[DType.float64]())


fn underflow_int[type: DType]() -> Bool:
    constrained[type.is_integral(), "comparison only valid on integral types"]()
    return min_finite[type]() - 1 > min_finite[type]()


fn underflow_fp[type: DType]() -> Bool:
    constrained[
        type.is_floating_point(),
        "comparison only valid on floating point types",
    ]()
    return min_finite[type]() - 1 == min_finite[type]()


def test_min_finite():
    assert_almost_equal(min_finite[DType.float32](), -3.4028235e38)
    assert_almost_equal(min_finite[DType.float64](), -1.7976931348623157e308)

    assert_true(underflow_int[DType.int8]())
    assert_true(underflow_int[DType.uint8]())
    assert_true(underflow_int[DType.int16]())
    assert_true(underflow_int[DType.uint16]())
    assert_true(underflow_int[DType.int32]())
    assert_true(underflow_int[DType.uint32]())
    assert_true(underflow_int[DType.int64]())
    assert_true(underflow_int[DType.uint64]())

    assert_true(underflow_fp[DType.float32]())
    assert_true(underflow_fp[DType.float64]())


def test_max_or_inf():
    assert_almost_equal(max_or_inf[DType.float32](), inf[DType.float32]())
    assert_almost_equal(max_or_inf[DType.float64](), inf[DType.float64]())


def test_min_or_neg_inf():
    assert_almost_equal(
        min_or_neg_inf[DType.float32](), neg_inf[DType.float32]()
    )
    assert_almost_equal(
        min_or_neg_inf[DType.float64](), neg_inf[DType.float64]()
    )


def test_neg_inf():
    assert_false(isfinite(neg_inf[DType.float32]()))
    assert_false(isfinite(neg_inf[DType.float64]()))
    assert_true(isinf(neg_inf[DType.float32]()))
    assert_true(isinf(neg_inf[DType.float64]()))
    assert_false(isnan(neg_inf[DType.float32]()))
    assert_false(isnan(neg_inf[DType.float64]()))
    assert_equal(-inf[DType.float32](), neg_inf[DType.float32]())
    assert_equal(-inf[DType.float64](), neg_inf[DType.float64]())


def test_nextafter():
    assert_true(isnan(nextafter(nan[DType.float32](), nan[DType.float32]())))
    assert_true(isinf(nextafter(inf[DType.float32](), inf[DType.float32]())))
    assert_true(isinf(nextafter(-inf[DType.float32](), -inf[DType.float32]())))
    assert_almost_equal(nextafter(Float64(0), Float64(0)), 0)
    assert_almost_equal(nextafter(Float64(0), Float64(1)), 5e-324)
    assert_almost_equal(nextafter(Float64(0), Float64(-1)), -5e-324)
    assert_almost_equal(nextafter(Float64(1), Float64(0)), 0.99999999999999988)
    assert_almost_equal(
        nextafter(Float64(-1), Float64(0)), -0.99999999999999988
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](0, 1)),
        SIMD[DType.float64, 2](0, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](1, 1)),
        SIMD[DType.float64, 2](5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](0, 1), SIMD[DType.float64, 2](-1, 1)),
        SIMD[DType.float64, 2](-5e-324, 1),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](1, 1), SIMD[DType.float64, 2](0, 0)),
        SIMD[DType.float64, 2](0.99999999999999988, 0.99999999999999988),
    )
    assert_almost_equal(
        nextafter(SIMD[DType.float64, 2](-1, -1), SIMD[DType.float64, 2](0, 0)),
        SIMD[DType.float64, 2](-0.99999999999999988, -0.99999999999999988),
    )


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
    test_get_accum_type()
    test_isfinite()
    test_isinf()
    test_isnan()
    test_max_finite()
    test_max_or_inf()
    test_min_finite()
    test_min_or_neg_inf()
    test_neg_inf()
    test_nextafter()
    test_ulp()
