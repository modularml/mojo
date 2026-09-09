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


def elementwise_max(lhs: SIMD, rhs: type_of(lhs), out result: type_of(lhs)):
    result = {}
    comptime for i in range(lhs.length):
        result[i] = lhs[i] if lhs[i] >= rhs[i] else rhs[i]


def main() raises:
    comptime simd_type = SIMD[.int64, 4]
    var v1 = simd_type(12, 0, 99, 77)
    var v2 = simd_type(0, 92, 11, 4)
    assert_equal(elementwise_max(v1, v2), simd_type(12, 92, 99, 77))

    var a: Array[Float64, 5] = [2, 4, 5, 7, 8]
    var b = Array[Float64, 4](fill=0)

    comptime for i in range(1, 5):
        b[i - 1] = a[i] + a[i - 1]

    var expected: Array[Float64, 4] = [6.0, 9.0, 12.0, 15.0]
    for i in range(expected.length):
        assert_equal(b[i], expected[i])
