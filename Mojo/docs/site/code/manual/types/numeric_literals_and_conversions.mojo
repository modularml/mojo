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
from std.testing import assert_equal


def main() raises:
    # start-numeric-literal-conversions
    var float1 = 3.3  # float1 is type Float64
    var float2: Float32 = 7.5
    var int1 = 5  # int1 is type Int
    var int2: Int8 = 4
    _, _, _, _ = float1, float2, int1, int2
    # end-numeric-literal-conversions

    # start-arbitrary-precision
    var arbitrary_precision = 3.0 * (4.0 / 3.0 - 1.0)
    # use a variable to force the following calculation to occur at runtime
    var three = 3.0
    var finite_precision = three * (4.0 / three - 1.0)
    print(arbitrary_precision, finite_precision)
    # end-arbitrary-precision
    assert_equal(arbitrary_precision, 1.0)
    assert_equal(finite_precision, 0.99999999999999978)

    # start-numeric-conversion-1
    var simd1 = SIMD[.float32, 4](2.2, 3.3, 4.4, 5.5)
    var simd2 = SIMD[.int16, 4](-1, 2, -3, 4)
    var simd3 = (
        simd1 * simd2.cast[DType.float32]()
    )  # Convert with cast() method
    print("simd3:", simd3)
    var simd4 = simd2 + SIMD[.int16, 4](simd1)  # Convert with SIMD constructor
    print("simd4:", simd4)
    # start-numeric-conversion-1
    assert_equal(simd3, SIMD[DType.float32, 4](-2.2, 6.6, -13.200001, 22.0))
    assert_equal(simd4, SIMD[DType.int16, 4](1, 5, 1, 9))

    # start-numeric-conversion-scalars
    var my_int: Int16 = 12  # SIMD[DType.int16, 1]
    var my_float: Float32 = 0.75  # SIMD[DType.float32, 1]
    var result = Float32(my_int) * my_float  # Result is SIMD[.float32, 1]
    print("Result:", result)
    # end-numeric-conversion-scalars
    assert_equal(result, 9.0)
