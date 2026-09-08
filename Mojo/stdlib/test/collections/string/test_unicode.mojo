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

from std.collections.string._unicode import _get_uppercase_mapping

from std.testing import assert_equal
from std.testing import TestSuite


def test_uppercase_conversion() raises:
    # a -> A
    var count1: Int
    count1, ref chars1 = _get_uppercase_mapping(Codepoint(97)).value()
    assert_equal(count1, 1)
    assert_equal(chars1[0], Codepoint(65))
    assert_equal(chars1[1], Codepoint(0))
    assert_equal(chars1[2], Codepoint(0))

    # ß -> SS
    var count2: Int
    count2, ref chars2 = _get_uppercase_mapping(
        Codepoint.from_u32(0xDF).value()
    ).value()
    assert_equal(count2, 2)
    assert_equal(chars2[0], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[1], Codepoint.from_u32(0x53).value())
    assert_equal(chars2[2], Codepoint(0))

    # ΐ -> Ϊ́
    var count3: Int
    count3, ref chars3 = _get_uppercase_mapping(
        Codepoint.from_u32(0x390).value()
    ).value()
    assert_equal(count3, 3)
    assert_equal(chars3[0], Codepoint.from_u32(0x0399).value())
    assert_equal(chars3[1], Codepoint.from_u32(0x0308).value())
    assert_equal(chars3[2], Codepoint.from_u32(0x0301).value())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
