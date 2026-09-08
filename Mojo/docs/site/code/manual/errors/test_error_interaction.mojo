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

from error_interaction import (
    wrapped_validate,
    validate_typed,
    validate_bare_raises,
)


def test_wrapped_validate_pass() raises:
    assert_equal(wrapped_validate(5), 5)


def test_wrapped_validate_fail() raises:
    with assert_raises(contains="cannot be negative"):
        _ = wrapped_validate(-5)


def test_wrapped_validate_field_access() raises:
    try:
        _ = wrapped_validate(-5)
    except e:
        assert_equal(e.field, "value")


def test_validate_typed_pass() raises:
    assert_equal(validate_typed(5), 5)


def test_validate_typed_fail() raises:
    with assert_raises(contains="cannot be negative"):
        _ = validate_typed(-5)


def test_bare_raises_erases_type() raises:
    # validate_bare_raises erases type info — only Error is caught
    with assert_raises(contains="cannot be negative"):
        _ = validate_bare_raises(-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
