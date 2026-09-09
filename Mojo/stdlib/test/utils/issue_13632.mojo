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

from std.testing import TestSuite, assert_equal


def sum_items(data: List[Int8]) -> Int:
    var sum: Int = 0
    for item in data:
        sum += Int(item)
    return sum


def make_abcd_vector() -> List[Int8]:
    return [97, 98, 99, 100]


def test_issue_13632() raises:
    var vec = make_abcd_vector()
    assert_equal(sum_items(vec), 394)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
