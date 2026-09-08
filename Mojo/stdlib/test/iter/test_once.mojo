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

from std.testing import TestSuite, assert_equal, assert_raises

from std.iter import once


def test_once() raises:
    var it = once(10)
    assert_equal(next(it), 10)
    with assert_raises():
        _ = next(it)


def test_once_owned() raises:
    var it = iter(once(10))
    assert_equal(next(it), 10)
    with assert_raises():
        _ = next(it)


def test_once_iter_copyable() raises:
    var it = once(10)
    var it_copy = iter(it)

    assert_equal(next(it), 10)
    with assert_raises():
        _ = next(it)

    assert_equal(next(it_copy), 10)
    with assert_raises():
        _ = next(it_copy)


def test_once_bounds() raises:
    var it = once(10)

    var lower, upper = it.bounds()
    assert_equal(lower, 1)
    assert_equal(upper, Optional(1))

    _ = next(it)

    lower, upper = it.bounds()
    assert_equal(lower, 0)
    assert_equal(upper, Optional(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
