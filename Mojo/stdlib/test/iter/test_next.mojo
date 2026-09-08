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

from std.testing import TestSuite, assert_equal, assert_true, assert_raises


def test_next() raises:
    var l = [1, 2, 3]

    var it = iter(l)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises():
        _ = next(it)  # raises StopIteration
    var l2 = ["hi", "hey", "hello"]
    var it2 = iter(l2)
    assert_equal(next(it2), "hi")
    assert_equal(next(it2), "hey")
    assert_equal(next(it2), "hello")
    with assert_raises():
        _ = next(it2)  # raises StopIteration


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
