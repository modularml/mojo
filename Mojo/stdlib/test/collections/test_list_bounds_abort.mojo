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
#
# Verifies that invalid `ContiguousSlice` indexing (out of bounds, reversed,
# or negative) aborts on `List` instead of silently clamping.
#
# ===----------------------------------------------------------------------=== #

from std.testing import _assert_aborts, TestSuite


def test_end_oob() raises:
    var l: List = [1, 2, 3]

    def trigger() raises {l} -> None:
        _ = l[0:100]

    _assert_aborts(
        trigger,
        contains="slice end index 100 is out of bounds, valid range is 0 to 3",
    )


def test_start_oob() raises:
    var l: List = [1, 2, 3]

    def trigger() raises {l} -> None:
        _ = l[100:200]

    _assert_aborts(
        trigger,
        contains=(
            "slice start index 100 is out of bounds, valid range is 0 to 3"
        ),
    )


def test_reversed() raises:
    var l: List = [1, 2, 3]

    def trigger() raises {l} -> None:
        _ = l[3:1]

    _assert_aborts(
        trigger,
        contains="slice start index 3 is greater than slice end index 1",
    )


def test_negative_start() raises:
    var l: List = [1, 2, 3]

    def trigger() raises {l} -> None:
        _ = l[-1:]

    _assert_aborts(
        trigger,
        contains="slice start index -1 is out of bounds, valid range is 0 to 3",
    )


def test_negative_end() raises:
    var l: List = [1, 2, 3]

    def trigger() raises {l} -> None:
        _ = l[:-1]

    _assert_aborts(
        trigger,
        contains="slice end index -1 is out of bounds, valid range is 0 to 3",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
