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

from std.sys import size_of

from std.testing import assert_equal, assert_false, assert_true, TestSuite

comptime uint_dtypes: List[DType] = [
    .uint8,
    .uint16,
    .uint32,
    .uint64,
    .uint128,
    .uint256,
]

comptime int_dtypes: List[DType] = [
    .int8,
    .int16,
    .int32,
    .int64,
    .int128,
    .int256,
]

comptime non_index_integral_dtypes = uint_dtypes + int_dtypes
comptime integral_dtypes = List(
    [DType.int, DType.uint]
) + non_index_integral_dtypes

comptime float_dtypes: List[DType] = [
    .float8_e3m4,
    .float8_e4m3fn,
    .float8_e4m3fnuz,
    .float8_e5m2,
    .float8_e5m2fnuz,
    .bfloat16,
    .float16,
    .float32,
    .float64,
]

comptime all_dtypes = List([DType.bool]) + integral_dtypes + float_dtypes


def test_equality() raises:
    assert_true(DType.float32 == .float32)
    assert_true(DType.float32 != .int32)
    assert_true(DType.float32 == .float32)
    assert_true(DType.float32 != .int32)


def test_stringable() raises:
    assert_equal(String(DType.bool), "bool")
    assert_equal(String(DType.int), "int")
    assert_equal(String(DType.uint), "uint")
    assert_equal(String(DType.int64), "int64")
    assert_equal(String(DType.float32), "float32")


def test_is_xxx() raises:
    def _is_category[
        test: def(DType) thin -> Bool,
        true_dtypes: List[DType],
    ]() raises:
        comptime for dt in all_dtypes:
            comptime res = dt in true_dtypes
            assert_equal(test(dt), res)

    # _is_category[DType.is_integral, integral_dtypes]()
    # _is_category[DType.is_floating_point, float_dtypes]()
    _is_category[DType.is_unsigned, List([DType.uint]) + uint_dtypes]()
    # _is_category[DType.is_signed, [DType.int] + int_dtypes + float_dtypes]()


def test_key_element() raises:
    var s = {DType.bool, DType.int64}
    assert_true(DType.int64 in s)
    assert_false(DType.float32 in s)


def test_from_str() raises:
    comptime dt = DType._from_str("bool")
    assert_equal(dt, DType.bool)

    assert_equal(DType._from_str("bool"), DType.bool)
    assert_equal(DType._from_str("DType.bool"), DType.bool)

    assert_equal(DType._from_str("int64"), DType.int64)
    assert_equal(DType._from_str("DType.int64"), DType.int64)

    assert_equal(DType._from_str("bfloat16"), DType.bfloat16)
    assert_equal(DType._from_str("DType.bfloat16"), DType.bfloat16)

    assert_false(DType._from_str("blahblah"))
    assert_false(DType._from_str("DType.blahblah"))

    comptime for dt in all_dtypes:
        assert_equal(DType._from_str(String(dt)), dt)


def test_mantissa_width() raises:
    """Mantissa widths must match each format's specification.

    Sub-byte dtypes are the interesting case: `bit_width_of` rounds them up to
    a whole byte, so a width derived from it reports a mantissa several bits
    too wide.
    """
    assert_equal(DType.mantissa_width[.float4_e2m1fn](), 1)
    assert_equal(DType.mantissa_width[.float6_e2m3fn](), 3)
    assert_equal(DType.mantissa_width[.float6_e3m2fn](), 2)
    assert_equal(DType.mantissa_width[.float8_e4m3fn](), 3)
    assert_equal(DType.mantissa_width[.float8_e5m2](), 2)
    assert_equal(DType.mantissa_width[.float16](), 10)
    assert_equal(DType.mantissa_width[.bfloat16](), 7)
    assert_equal(DType.mantissa_width[.float32](), 23)
    assert_equal(DType.mantissa_width[.float64](), 52)


def test_float6() raises:
    """The OCP MX FP6 encodings, per the microscaling specification."""
    assert_equal(DType.exponent_width[.float6_e2m3fn](), 2)
    assert_equal(DType.exponent_width[.float6_e3m2fn](), 3)

    assert_equal(DType.max_exponent[.float6_e2m3fn](), 2)
    assert_equal(DType.max_exponent[.float6_e3m2fn](), 4)

    # Storage formats with no native arithmetic, like float4_e2m1fn.
    assert_false(DType.float6_e2m3fn.is_numeric())
    assert_false(DType.float6_e3m2fn.is_numeric())
    assert_true(DType.float6_e2m3fn.is_floating_point())
    assert_true(DType.float6_e3m2fn.is_floating_point())

    assert_equal(String(DType.float6_e2m3fn), "float6_e2m3fn")
    assert_equal(String(DType.float6_e3m2fn), "float6_e3m2fn")
    assert_equal(DType._from_str("float6_e2m3fn"), DType.float6_e2m3fn)
    assert_equal(DType._from_str("float6_e3m2fn"), DType.float6_e3m2fn)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
