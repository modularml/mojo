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

from std.collections.string._unicode import (
    _get_uppercase_mapping,
    BIGGEST_UNICODE_CODEPOINT,
)

from std.testing import TestSuite, assert_equal, assert_raises


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


def test_codepoint_parsing() raises:
    var buf: List[UInt8] = [0]

    # ASCII
    for i in range(0x7F + 1):
        buf[0] = UInt8(i)
        assert_equal(String(from_codepoints=buf), chr(Int(i)))
        assert_equal(String(from_codepoints=buf, errors="replace"), chr(Int(i)))

    # ISO-8859-1 aka. Latin-1
    for i in range(0x7F + 1, 0xFF + 1):
        buf[0] = UInt8(i)
        assert_equal(String(from_codepoints=buf), chr(Int(i)))
        assert_equal(String(from_codepoints=buf, errors="replace"), chr(Int(i)))

    # UInt16 range
    var buf16: List[UInt16] = [0]
    for i in range(UInt16(0xFF + 1), UInt16.MAX):
        buf16[0] = i
        if UInt16(0xD800) <= i <= 0xDFFF:  # UTF-16 surrogate pairs
            with assert_raises():
                _ = String(from_codepoints=buf16)
            assert_equal(String(from_codepoints=buf16, errors="replace"), "�")
        else:
            assert_equal(String(from_codepoints=buf16), chr(Int(i)))
            assert_equal(
                String(from_codepoints=buf16, errors="replace"), chr(Int(i))
            )

    # UInt32 range
    var buf32: List[UInt32] = [0]
    for i in range(UInt32(UInt16.MAX), BIGGEST_UNICODE_CODEPOINT + 1):
        buf32[0] = i
        assert_equal(String(from_codepoints=buf32), chr(Int(i)))
        assert_equal(
            String(from_codepoints=buf32, errors="replace"), chr(Int(i))
        )

    # test invalid range
    var buf64: List[UInt64] = [UInt64(BIGGEST_UNICODE_CODEPOINT) + 1]
    with assert_raises():
        _ = String(from_codepoints=buf64)
    assert_equal(String(from_codepoints=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX)]
    with assert_raises():
        _ = String(from_codepoints=buf64)
    assert_equal(String(from_codepoints=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX) + 2]
    with assert_raises():
        _ = String(from_codepoints=buf64)
    assert_equal(String(from_codepoints=buf64, errors="replace"), "�")
    buf64 = [UInt64.MAX - 2]
    with assert_raises():
        _ = String(from_codepoints=buf64)
    assert_equal(String(from_codepoints=buf64, errors="replace"), "�")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
