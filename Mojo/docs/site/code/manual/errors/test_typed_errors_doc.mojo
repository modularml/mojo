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

from typed_errors import ValidationError, validate_username


def test_valid_username() raises:
    assert_equal(validate_username("alice"), "alice")


def test_empty_username_raises() raises:
    with assert_raises(contains="cannot be empty"):
        _ = validate_username("")


def test_short_username_raises() raises:
    with assert_raises(contains="must be at least 3 characters"):
        _ = validate_username("ab")


def test_field_access() raises:
    try:
        _ = validate_username("")
    except e:
        assert_equal(e.field, "username")
        assert_equal(e.reason, "cannot be empty")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
