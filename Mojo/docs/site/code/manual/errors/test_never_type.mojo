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
from std.testing import assert_equal, assert_raises, TestSuite

from never_type import panic, get_value_or_panic, safe_add


def test_panic_always_raises() raises:
    with assert_raises(contains="boom"):
        panic("boom")


def test_get_value_with_some() raises:
    assert_equal(get_value_or_panic(Optional(42)), 42)


def test_get_value_with_none() raises:
    with assert_raises(contains="value is missing"):
        _ = get_value_or_panic(Optional[Int]())


def test_safe_add() raises:
    assert_equal(safe_add(3, 4), 7)
    assert_equal(safe_add(0, 0), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
