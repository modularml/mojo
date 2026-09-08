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

from test_utils import ulp_distance
from std.testing import assert_true, TestSuite


def test_ulp_distance() raises:
    var a: Float64 = 1.0
    var b: Float64 = 1.0
    var ulp = ulp_distance(a, b)
    assert_true(ulp == 0)

    a = 1.0
    b = 1.0000000000000002
    ulp = ulp_distance(a, b)
    assert_true(ulp == 1)

    a = 1.0
    b = 1.0000000000000022
    ulp = ulp_distance(a, b)
    assert_true(ulp == 10)

    var a_f32 = Float32(1.0)
    var b_f32 = Float32(1.0000001)
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 1)

    a_f32 = Float32(-1.0)
    b_f32 = Float32(-1.0000001)
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 1)

    a_f32 = Float32(-1)
    b_f32 = Float32(4)
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 2**31)

    a_f32 = 1.0
    b_f32 = 1.0000014
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 12)

    var a_bf16 = UInt16(0x3F80)
    var b_bf16 = UInt16(0x3F8D)
    ulp = ulp_distance(a_bf16, b_bf16)
    assert_true(ulp == 13)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
