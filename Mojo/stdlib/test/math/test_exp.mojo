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

from std.math.math import _Expable, exp
from std.random import randn_float64, seed
from std.sys import CompilationTarget

from test_utils import libm_call
from std.testing import assert_almost_equal, assert_equal, TestSuite


def test_exp_bfloat16() raises:
    assert_equal(exp(BFloat16(2.0)), 7.375)
    assert_equal(exp(5.0) * exp(6.0), exp(5.0 + 6.0))
    assert_equal(exp(5.0) / exp(2.0), exp(5.0 - 2.0))


def test_exp_float16() raises:
    assert_almost_equal(exp(Float16(-0.1)), 0.9047)
    assert_almost_equal(exp(Float16(0.1)), 1.105)
    assert_almost_equal(exp(Float16(2)), 7.389)
    assert_equal(String(exp(Float16(89))), "inf")
    assert_equal(String(exp(Float16(108.5230))), "inf")


def test_exp_float32() raises:
    assert_almost_equal(exp(Float32(-0.1)), 0.90483)
    assert_almost_equal(exp(Float32(0.1)), 1.10517)
    assert_almost_equal(exp(Float32(2)), 7.38905)
    assert_equal(String(exp(Float32(89))), "inf")
    assert_equal(String(exp(Float32(108.5230))), "inf")


def test_exp_float64() raises:
    assert_almost_equal(exp(Float64(-0.1)), 0.90483)
    assert_almost_equal(exp(Float64(0.1)), 1.10517)
    assert_almost_equal(exp(Float64(2)), 7.38905)
    # FIXME (40568) should remove str
    assert_equal(String(exp(Float64(89))), String(4.4896128193366053e38))
    assert_equal(String(exp(Float64(108.5230))), String(1.3518859659123633e47))


@always_inline
def exp_libm[
    dtype: DType, simd_width: SIMDLength
](arg: SIMD[dtype, simd_width]) raises -> SIMD[dtype, simd_width]:
    return libm_call["expf", "exp"](arg)


def _test_exp_libm[dtype: DType]() raises where dtype.is_floating_point():
    seed(0)
    comptime N = 8192
    for _i in range(N):
        var x = randn_float64(0, 9.0).cast[dtype]()
        assert_almost_equal(
            exp(x), exp_libm(x), msg=String("for the input ", x)
        )


def test_exp_libm() raises:
    _test_exp_libm[.float32]()
    _test_exp_libm[.float64]()


@fieldwise_init
struct Float32Expable(Equatable, Writable, _Expable):
    """This is a test struct that implements the Expable trait for Float32."""

    var x: Float32

    def __exp__(self) -> Self:
        return Self(exp(self.x))

    def __eq__(self, other: Self) -> Bool:
        return self.x == other.x

    def __ne__(self, other: Self) -> Bool:
        return self.x != other.x

    def write_to(self, mut writer: Some[Writer]):
        t"Float32Expable({self.x})".write_to(writer)


@fieldwise_init
struct FakeExpable(Equatable, Writable, _Expable):
    """Test struct using default reflection-based __eq__."""

    var x: Int

    def __exp__(self) -> Self:
        return Self(99)

    # Uses default reflection-based __eq__ from Equatable trait

    def write_to(self, mut writer: Some[Writer]):
        t"FakeExpable({self.x})".write_to(writer)


def test_exapble_trait() raises:
    assert_equal(exp(Float32Expable(1.0)), Float32Expable(exp(Float32(1.0))))
    assert_equal(exp(FakeExpable(1)), FakeExpable(99))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
