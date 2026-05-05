# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from std.collections.string._utf16 import _decode_utf16

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #
# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_utf16_parsing() raises:
    var buf: List[UInt16] = []
    assert_equal(_decode_utf16(from_utf16=buf), "")
    buf = [65, 97, 33, 945, 10175]
    assert_equal(_decode_utf16(from_utf16=buf), "Aa!α➿")
    buf = [0x20AC]
    assert_equal(_decode_utf16(from_utf16=buf), "€")
    buf = [0xFFFD]
    assert_equal(_decode_utf16(from_utf16=buf), "�")
    buf = [0xD83D, 0xDD25]
    assert_equal(_decode_utf16(from_utf16=buf), "🔥")
    buf = [0xD801, 0xDC37]
    assert_equal(_decode_utf16(from_utf16=buf), chr(0x10437))
    buf = [0xD852, 0xDF62]
    assert_equal(_decode_utf16(from_utf16=buf), chr(0x24B62))

    buf = [0]
    for i in range(0xD800):
        buf[0] = UInt16(i)
        assert_equal(String(from_utf16=buf), chr(i))
        assert_equal(String(from_utf16=buf, errors="replace"), chr(i))

    # test unpaired surrogates
    for i in range(0xD800, 0xDFFF + 1):
        buf[0] = UInt16(i)
        with assert_raises():
            _ = String(from_utf16=buf)
        assert_equal(String(from_utf16=buf, errors="replace"), "�")

    for i in range(0xDFFF + 1, Int(UInt16.MAX) + 1):
        buf[0] = UInt16(i)
        assert_equal(String(from_utf16=buf), chr(i))
        assert_equal(String(from_utf16=buf, errors="replace"), chr(i))

    # test invalid range
    var buf64: List[UInt64] = [UInt64(UInt16.MAX) + 1]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX)]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64(UInt32.MAX) + 2]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")
    buf64 = [UInt64.MAX - 2]
    with assert_raises():
        _ = String(from_utf16=buf64)
    assert_equal(String(from_utf16=buf64, errors="replace"), "�")



def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
