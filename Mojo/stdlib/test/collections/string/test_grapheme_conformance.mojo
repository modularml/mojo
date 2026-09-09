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
"""Unicode grapheme cluster conformance test.

Runs all official GraphemeBreakTest.txt test vectors from the Unicode
consortium to verify UAX #29 compliance. These test the segmentation
*algorithm* (state machine transitions), not the property tables — for
example, verifying that a ZWJ sequence between two emoji stays together,
or that regional indicator pairs form flag graphemes.
"""

from std.builtin.globals import global_constant
from std.testing import assert_equal, TestSuite

from _grapheme_conformance_data import GRAPHEME_CONFORMANCE_DATA


def _string_from_data(
    data: Span[UInt32, _], offset: Int, num_cps: Int
) -> String:
    """Build a String from codepoints in the test data array."""
    var s = String()
    for i in range(num_cps):
        s += chr(Int(data[offset + i]))
    return s


def test_unicode_grapheme_break_conformance() raises:
    """Run all official GraphemeBreakTest.txt test vectors."""
    var data = Span(global_constant[GRAPHEME_CONFORMANCE_DATA]())
    var idx = 0
    var test_num = 0

    while idx < len(data):
        var expected_count = Int(data[idx])
        var num_cps = Int(data[idx + 1])
        idx += 2

        var s = _string_from_data(data, idx, num_cps)
        idx += num_cps
        test_num += 1

        assert_equal(
            s.count_graphemes(),
            expected_count,
            msg=String(
                t"test vector {test_num}: expected {expected_count}"
                t" graphemes for {num_cps} codepoints"
            ),
        )


def test_unicode_grapheme_reverse_matches_forward() raises:
    """Every conformance vector, iterated in reverse, produces the same
    clusters as forward iteration reversed."""
    var data = Span(global_constant[GRAPHEME_CONFORMANCE_DATA]())
    var idx = 0
    var test_num = 0

    while idx < len(data):
        var num_cps = Int(data[idx + 1])
        idx += 2
        var s = _string_from_data(data, idx, num_cps)
        idx += num_cps
        test_num += 1

        var fwd = List[String]()
        for g in s.graphemes():
            fwd.append(String(g))

        var rev = List[String]()
        for g in s.graphemes_reversed():
            rev.append(String(g))

        assert_equal(
            len(fwd),
            len(rev),
            msg=String(
                t"test vector {test_num}: forward/reverse length mismatch"
            ),
        )
        for i in range(len(fwd)):
            assert_equal(
                fwd[len(fwd) - 1 - i],
                rev[i],
                msg=String(
                    t"test vector {test_num} position {i}: forward/reverse"
                    t" cluster mismatch"
                ),
            )


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
