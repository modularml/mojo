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

from std.os import abort
from std.testing import (
    assert_raises,
    assert_equal,
    assert_true,
    assert_false,
    TestSuite,
)
from std.testing.prop.random import Rng

comptime FLOAT_DTYPES = [
    DType.float16,
    DType.bfloat16,
    DType.float32,
    DType.float64,
]


def test_rng_xoshiro_float() raises:
    var rng = Rng(seed=1234)
    for _ in range(100):
        var f = rng._xoshiro_float()
        assert_true(f >= 0.0)
        assert_true(f <= 1.0)


def test_rng_rand_bool() raises:
    var rng = Rng(seed=1234)
    for _ in range(100):
        assert_true(rng.rand_bool(true_probability=1.0))
        assert_false(rng.rand_bool(true_probability=0.0))


def test_rng_rand_scalar() raises:
    def test_dtype[dtype: DType](min: Scalar[dtype], max: Scalar[dtype]) raises:
        var rng = Rng(seed=1234)
        for _ in range(100):
            var value = rng.rand_scalar[dtype](min=min, max=max)
            assert_true(value >= min)
            assert_true(value <= max)

    comptime for dtype in [
        DType.uint,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
        DType.uint128,
        DType.uint256,
        DType.int,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int128,
        DType.int256,
        DType.float16,
        DType.bfloat16,
        DType.float32,
        DType.float64,
    ]:
        comptime scalar = Scalar[dtype]

        test_dtype[dtype](scalar.MIN_FINITE, scalar.MAX_FINITE)

        comptime if dtype.is_signed():
            test_dtype[dtype](scalar(-10), scalar(10))
        else:
            test_dtype[dtype](scalar(10), scalar(20))


def test_rng_rand_scalar_wide_covers_full_range() raises:
    """Wide integer dtypes must produce magnitudes outside the UInt64 range."""

    def check[dtype: DType]() raises {}:
        var rng = Rng(seed=1234)
        var threshold = Scalar[dtype](UInt64.MAX)
        var saw_above = False
        var saw_below_zero = False
        for _ in range(2000):
            var v = rng.rand_scalar[dtype]()
            comptime if dtype.is_signed():
                if v < Scalar[dtype](0):
                    saw_below_zero = True
                if v > threshold or v < -threshold:
                    saw_above = True
            else:
                if v > threshold:
                    saw_above = True

        assert_true(
            saw_above,
            "expected at least one value with magnitude > UInt64.MAX",
        )
        comptime if dtype.is_signed():
            assert_true(saw_below_zero, "expected at least one negative value")

    check[DType.uint128]()
    check[DType.uint256]()
    check[DType.int128]()
    check[DType.int256]()


def test_rng_rand_scalar_raises() raises:
    with assert_raises(contains="invalid min/max"):
        var rng = Rng(seed=1234)
        var _ = rng.rand_scalar[.int32](min=10, max=5)


def test_rng_rand_scalar_float_covers_range_interior() raises:
    """A bounded float range must produce values spread across the range, not
    values pinned to its endpoints."""

    def check[dtype: DType](min: Scalar[dtype], max: Scalar[dtype]) raises:
        comptime draws = 1000
        var rng = Rng(seed=1234)
        var inside = 0
        for _ in range(draws):
            var v = rng.rand_scalar[dtype](min=min, max=max)
            if min < v < max:
                inside += 1
        assert_true(
            inside > draws // 2,
            String(
                t"expected most {dtype} values strictly inside [{min}, {max}],"
                t" got {inside} of {draws}"
            ),
        )

    comptime for dtype in FLOAT_DTYPES:
        comptime scalar = Scalar[dtype]
        check[dtype](scalar(10), scalar(20))
        check[dtype](scalar(-20), scalar(-10))
        check[dtype](scalar(0), scalar(1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
