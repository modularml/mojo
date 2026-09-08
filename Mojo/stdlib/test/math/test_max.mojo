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

from std.testing import TestSuite
from std.testing import assert_equal


def test_max() raises:
    var expected_result = SIMD[.bool, 4](True, True, False, True)
    var actual_result = max(
        SIMD[.bool, 4](
            True,
            True,
            False,
            False,
        ),
        SIMD[.bool, 4](False, True, False, True),
    )

    assert_equal(actual_result, expected_result)


def test_max_scalar() raises:
    assert_equal(max(Bool(True), Bool(False)), Bool(True))
    assert_equal(max(Bool(False), Bool(True)), Bool(True))
    assert_equal(max(Bool(False), Bool(False)), Bool(False))
    assert_equal(max(Bool(True), Bool(True)), Bool(True))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
