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

from std.hex import hex_encode, hex_decode

from std.testing import assert_equal, assert_raises
from std.testing import TestSuite


def test_hex_encode() raises:
    var random_bytes: List[Byte] = [
        0x01,
        0x23,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xFF,
    ]
    assert_equal(hex_encode(random_bytes[:]), "012356789abcdeff")


def test_hex_decode() raises:
    var random_bytes: List[Byte] = [
        0x01,
        0x23,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xFF,
    ]
    assert_equal(hex_decode("012356789abcdeff"), random_bytes)

    var random_bytes_fixed: InlineArray[Byte, 8] = [
        0x01,
        0x23,
        0x56,
        0x78,
        0x9A,
        0xBC,
        0xDE,
        0xFF,
    ]
    assert_equal(hex_decode[8]("012356789abcdeff"), random_bytes_fixed)


def test_ivalid_hex_decode() raises:
    with assert_raises():
        _ = hex_decode("abc")

    with assert_raises():
        _ = hex_decode[4]("00ff")  # too short

    with assert_raises():
        _ = hex_decode[4]("00ffff0011")  # too long

    with assert_raises():
        _ = hex_decode("0g")  # invalid character

    with assert_raises():
        _ = hex_decode("0Ƹ")  # invalid character


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
