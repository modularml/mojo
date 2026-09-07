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
from std.testing import assert_raises, assert_equal, assert_true, TestSuite

from typed_errors_testing import (
    ValidationError,
    ConfigError,
    validate_username,
    validate_age,
    load_config,
)


# =============================================================================
# Tests for assert_raises with typed errors
# =============================================================================


def test_assert_raises_basic() raises:
    """Test assert_raises catches typed errors."""
    with assert_raises():
        validate_username("")


def test_assert_raises_contains_message() raises:
    """Test assert_raises(contains=...) matches error messages."""
    with assert_raises(contains="cannot be empty"):
        validate_username("")


def test_assert_raises_contains_reason() raises:
    """Test assert_raises matches partial reason."""
    with assert_raises(contains="at least 3"):
        validate_username("ab")


def test_assert_raises_contains_field() raises:
    """Test assert_raises matches field name in output."""
    with assert_raises(contains="field: username"):
        validate_username("")


def test_assert_raises_config_error() raises:
    """Test assert_raises with ConfigError."""
    with assert_raises(contains="key not found"):
        _ = load_config("missing")


def test_assert_raises_config_invalid() raises:
    """Test assert_raises with different ConfigError reason."""
    with assert_raises(contains="invalid value format"):
        _ = load_config("invalid")


# =============================================================================
# Tests for type verification via field access
# =============================================================================


def test_verify_validation_error_fields() raises:
    """Test verifying ValidationError type via field access."""
    var field = ""
    var reason = ""

    try:
        validate_username("")
    except e:
        field = e.field
        reason = e.reason

    assert_equal(field, "username")
    assert_equal(reason, "cannot be empty")


def test_verify_config_error_fields() raises:
    """Test verifying ConfigError type via field access."""
    var key = ""
    var message = ""

    try:
        _ = load_config("missing")
    except e:
        key = e.key
        message = e.message

    assert_equal(key, "missing")
    assert_equal(message, "key not found")


def test_verify_age_validation() raises:
    """Test age validation error fields."""
    var field = ""
    var reason = ""

    try:
        validate_age(-1)
    except e:
        field = e.field
        reason = e.reason

    assert_equal(field, "age")
    assert_equal(reason, "cannot be negative")


# =============================================================================
# Tests for workaround pattern (flag variable)
# =============================================================================


def test_flag_pattern_error_raised() raises:
    """Test flag pattern confirms error was raised."""
    var raised = False
    try:
        validate_username("")
    except e:
        raised = True
        _ = e.field  # Proves type

    assert_true(raised, "Error should be raised")


def test_flag_pattern_no_error() raises:
    """Test flag pattern when no error is raised."""
    var raised = False
    try:
        validate_username("validname")
    except e:
        raised = True

    assert_true(not raised, "No error should be raised")


def test_flag_pattern_multiple_conditions() raises:
    """Test flag pattern with multiple error conditions."""
    var empty_raised = False
    var short_raised = False

    try:
        validate_username("")
    except e:
        empty_raised = True
        assert_equal(e.reason, "cannot be empty")

    try:
        validate_username("ab")
    except e:
        short_raised = True
        assert_equal(e.reason, "must be at least 3 characters")

    assert_true(empty_raised, "Empty username should raise")
    assert_true(short_raised, "Short username should raise")


# =============================================================================
# Tests for successful operations (no error)
# =============================================================================


def test_valid_username() raises:
    """Test valid username does not raise."""
    # Should complete without error
    validate_username("validuser")


def test_valid_age() raises:
    """Test valid age does not raise."""
    validate_age(25)


def test_valid_config() raises:
    """Test valid config key returns value."""
    var value = load_config("database")
    assert_equal(value, "value_for_database")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
