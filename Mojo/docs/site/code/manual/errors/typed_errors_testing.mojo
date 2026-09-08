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
"""Demonstrates testing typed errors with the Mojo testing framework.

Key findings:
- assert_raises() works correctly with typed errors
- assert_raises(contains=...) matches typed error message strings
- Cannot assert specific error TYPE, only that some error was raised
- Cannot mix `Error`-raising functions inside typed error try blocks
- Workaround: Use flag variables and verify fields after try block

Recommended patterns:
1. Use assert_raises(contains=...) for message verification
2. Manually catch and verify fields for type verification
3. Use flag variables to avoid mixing error types in try blocks
"""

from std.testing import assert_raises, assert_equal, assert_true


@fieldwise_init
struct ValidationError(Copyable, Writable):
    """A typed error for validation failures."""

    var field: String
    var reason: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ValidationError: ", self.reason, " (field: ", self.field, ")"
        )


@fieldwise_init
struct ConfigError(Copyable, Writable):
    """A typed error for configuration problems."""

    var key: String
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ConfigError[", self.key, "]: ", self.message)


# =============================================================================
# Functions that raise typed errors
# =============================================================================


def validate_username(name: String) raises ValidationError:
    """Validate a username."""
    if not name:
        raise ValidationError(field="username", reason="cannot be empty")
    if name.count_codepoints() < 3:
        raise ValidationError(
            field="username", reason="must be at least 3 characters"
        )


def validate_age(age: Int) raises ValidationError:
    """Validate an age value."""
    if age < 0:
        raise ValidationError(field="age", reason="cannot be negative")
    if age > 150:
        raise ValidationError(field="age", reason="must be realistic")


def load_config(key: String) raises ConfigError -> String:
    """Load a configuration value."""
    if key == "missing":
        raise ConfigError(key=key, message="key not found")
    if key == "invalid":
        raise ConfigError(key=key, message="invalid value format")
    return "value_for_" + key


# =============================================================================
# Pattern 1: Using assert_raises with typed errors
# =============================================================================


def demonstrate_assert_raises() raises:
    """Shows assert_raises works with typed errors."""
    print("=== Pattern 1: assert_raises ===")

    # Basic assertion - just check that an error is raised
    print("Testing empty username...")
    with assert_raises():
        validate_username("")
    print("  Passed: Error was raised")

    # With contains - check error message
    print("Testing short username...")
    with assert_raises(contains="at least 3 characters"):
        validate_username("ab")
    print("  Passed: Message contains expected text")

    # Can match field names in error output
    print("Testing field name in message...")
    with assert_raises(contains="field: username"):
        validate_username("")
    print("  Passed: Field name found in message")


# =============================================================================
# Pattern 2: Verifying error type via field access
# =============================================================================


def demonstrate_type_verification() raises:
    """Shows how to verify specific error type was raised."""
    print("\n=== Pattern 2: Type verification ===")

    # Capture error fields to verify type
    var error_field = ""
    var error_reason = ""

    try:
        validate_username("")
    except e:
        # Field access proves this is ValidationError
        error_field = e.field
        error_reason = e.reason

    # Verify after try block (avoid mixing error types)
    assert_equal(error_field, "username")
    assert_equal(error_reason, "cannot be empty")
    print("Verified ValidationError with correct fields")


# =============================================================================
# Pattern 3: Testing multiple error scenarios
# =============================================================================


def demonstrate_multiple_scenarios() raises:
    """Shows testing various error conditions."""
    print("\n=== Pattern 3: Multiple scenarios ===")

    # Test different error conditions for the same function
    print("Testing username validation scenarios...")

    with assert_raises(contains="cannot be empty"):
        validate_username("")

    with assert_raises(contains="at least 3 characters"):
        validate_username("ab")

    print("  All scenarios passed")

    # Test different functions with different error types
    print("Testing config loading...")

    with assert_raises(contains="key not found"):
        _ = load_config("missing")

    with assert_raises(contains="invalid value format"):
        _ = load_config("invalid")

    print("  All scenarios passed")


# =============================================================================
# Pattern 4: Workaround for mixing error types
# =============================================================================


def demonstrate_workaround() raises:
    """Shows workaround for the error type mixing limitation."""
    print("\n=== Pattern 4: Error type workaround ===")

    # Problem: Cannot call assert_true inside typed error try block
    # Solution: Use flag variable

    var raised = False
    try:
        validate_age(-1)
    except e:
        raised = True
        # Can still verify fields here
        _ = e.field  # Would fail to compile if not ValidationError

    # Check flag outside try block
    assert_true(raised, "Expected ValidationError")
    print("Error was raised and caught correctly")


# =============================================================================
# Main demonstration
# =============================================================================


def main() raises:
    demonstrate_assert_raises()
    demonstrate_type_verification()
    demonstrate_multiple_scenarios()
    demonstrate_workaround()
    print("\nAll demonstrations completed successfully!")
