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

from std.sys.info import bit_width_of

from std.bit.mask import is_negative, splat
from std.testing import assert_equal
from std.testing import TestSuite


def test_is_negative() raises:
    comptime dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
    )
    comptime widths = (1, 2, 4, 8)

    comptime for i in range(len(dtypes)):
        comptime D = rebind[DType](dtypes[i])
        var last_value = 2 ** (bit_width_of[D]() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        comptime for j in range(len(widths)):
            comptime S = SIMD[D, rebind[Int](widths[j])]

            for k in values:
                assert_equal(S(-1), is_negative(S(-k)))
                assert_equal(S(0), is_negative(S(k)))


def test_splat() raises:
    comptime dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
        DType.uint8,
        DType.uint16,
        DType.uint32,
        DType.uint64,
    )
    comptime widths = (1, 2, 4, 8)

    comptime for i in range(len(dtypes)):
        comptime D = rebind[DType](dtypes[i])

        comptime for j in range(len(widths)):
            comptime w = rebind[Int](widths[j])
            comptime B = SIMD[.bool, w]
            assert_equal(SIMD[D, w](-1), splat[D](B(fill=True)))
            assert_equal(SIMD[D, w](0), splat[D](B(fill=False)))


def test_compare() raises:
    comptime dtypes = (
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.int,
    )
    comptime widths = (1, 2, 4, 8)

    comptime for i in range(len(dtypes)):
        comptime D = rebind[DType](dtypes[i])
        var last_value = 2 ** (bit_width_of[D]() - 1) - 1
        var values = [1, 2, last_value - 1, last_value]

        comptime for j in range(len(widths)):
            comptime S = SIMD[D, rebind[Int](widths[j])]

            for k in values:
                var s_k = S(k)
                var s_k_1 = S(k - 1)
                assert_equal(S(-1), splat[D](s_k.eq(s_k)))
                assert_equal(S(-1), splat[D]((-s_k).eq(-s_k)))
                assert_equal(S(-1), splat[D](s_k.ne(s_k_1)))
                assert_equal(S(-1), splat[D]((-s_k).ne(s_k_1)))
                assert_equal(S(-1), splat[D](s_k.gt(s_k_1)))
                assert_equal(S(-1), splat[D](s_k_1.gt(-s_k)))
                assert_equal(S(-1), splat[D]((-s_k).ge(-s_k)))
                assert_equal(S(-1), splat[D]((-s_k).lt(s_k_1)))
                assert_equal(S(-1), splat[D]((-s_k).le(-s_k)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
