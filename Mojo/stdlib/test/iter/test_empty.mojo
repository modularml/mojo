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

from std.testing import TestSuite, assert_raises, assert_equal

from std.iter import empty


def test_empty() raises:
    var it = empty[Int]()
    with assert_raises():
        _ = next(it)


def test_empty_owned() raises:
    var it = iter(empty[Int]())
    with assert_raises():
        _ = next(it)


def test_empty_bounds() raises:
    var it = empty[Int]()

    var lower, upper = it.bounds()
    assert_equal(lower, 0)
    assert_equal(upper, Optional(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
