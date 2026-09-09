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
"""Demonstrates edge cases and patterns for typed error handling.

Key findings:
- Multiple except clauses are NOT supported (use nested try blocks instead)
- Python-style `except ErrorType as e:` syntax is NOT supported
- Errors propagate correctly through multiple function calls
- Re-raising works with `raise e^` (transfers ownership)
- Error suppression works by catching and not re-raising
- Compiler warns about unreachable except blocks

Compiler behaviors documented:
- `except ErrorType as e:` -> "expected ':' after 'except'"
- Try block with non-raising code -> "except logic is unreachable"
"""


@fieldwise_init
struct NetworkError(Copyable, Writable):
    """Error for network operations."""

    var code: Int
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("NetworkError[", self.code, "]: ", self.message)


@fieldwise_init
struct ParseError(Copyable, Writable):
    """Error for parsing operations."""

    var position: Int
    var expected: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ParseError at ", self.position, ": expected ", self.expected
        )


@fieldwise_init
struct ApplicationError(Copyable, Writable):
    """Unified error type for application-level errors."""

    var source: String
    var details: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ApplicationError[", self.source, "]: ", self.details)

    @staticmethod
    def from_network(e: NetworkError) -> Self:
        """Convert a NetworkError to ApplicationError."""
        return Self(source="network", details=String(e))

    @staticmethod
    def from_parse(e: ParseError) -> Self:
        """Convert a ParseError to ApplicationError."""
        return Self(source="parse", details=String(e))


# =============================================================================
# Pattern 1: Nested try blocks for handling different error types
# =============================================================================


def fetch_data() raises NetworkError -> String:
    """Simulate fetching data that may fail."""
    raise NetworkError(code=404, message="Resource not found")


def parse_data(data: String) raises ParseError -> Int:
    """Simulate parsing data that may fail."""
    raise ParseError(position=0, expected="integer")


def fetch_and_parse() raises ApplicationError -> Int:
    """Handle different error types using nested try blocks.

    Since Mojo only supports one except clause per try block,
    use nested blocks to handle different error types.
    """
    var data: String
    try:
        data = fetch_data()
    except e:
        raise ApplicationError.from_network(e)

    try:
        return parse_data(data)
    except e:
        raise ApplicationError.from_parse(e)


# =============================================================================
# Pattern 2: Error propagation through call chain
# =============================================================================


def level3() raises NetworkError:
    """Deepest level - raises the error."""
    raise NetworkError(code=500, message="Server error")


def level2() raises NetworkError:
    """Middle level - propagates error unchanged."""
    level3()


def level1() raises NetworkError:
    """Top level - propagates error unchanged."""
    level2()


# =============================================================================
# Pattern 3: Re-raising errors
# =============================================================================


def reraise_with_logging() raises NetworkError -> String:
    """Catch, log, and re-raise the same error."""
    try:
        return fetch_data()
    except e:
        print("Logging before re-raise:", e)
        raise e^  # Use ^ to transfer ownership


def reraise_as_different_type() raises ApplicationError -> String:
    """Catch one type and raise a different type."""
    try:
        return fetch_data()
    except e:
        raise ApplicationError.from_network(e)


def reraise_modified() raises NetworkError -> String:
    """Catch, modify, and raise a new error."""
    try:
        return fetch_data()
    except e:
        raise NetworkError(code=e.code, message="[Modified] " + e.message)


# =============================================================================
# Pattern 4: Error suppression
# =============================================================================


def suppress_error() -> String:
    """Catch an error and return a default value instead of re-raising."""
    try:
        return fetch_data()
    except e:
        print("Suppressed error:", e)
        return "default value"


def suppress_and_continue():
    """Catch an error and continue execution."""
    try:
        _ = fetch_data()
    except e:
        print("Ignored error:", e)
    print("Continuing after suppressed error")


# =============================================================================
# Demonstration
# =============================================================================


def main():
    print("=== Pattern 1: Nested try blocks ===")
    try:
        _ = fetch_and_parse()
    except e:
        print("Unified error:", e)
        print("  Source:", e.source)

    print("\n=== Pattern 2: Error propagation ===")
    try:
        level1()
    except e:
        print("Propagated through 3 levels:", e)

    print("\n=== Pattern 3: Re-raising ===")

    print("\n--- With logging ---")
    try:
        _ = reraise_with_logging()
    except e:
        print("After re-raise:", e)

    print("\n--- As different type ---")
    try:
        _ = reraise_as_different_type()
    except e:
        print("Converted:", e)

    print("\n--- Modified ---")
    try:
        _ = reraise_modified()
    except e:
        print("Modified:", e)

    print("\n=== Pattern 4: Error suppression ===")

    print("\n--- Return default ---")
    var result = suppress_error()
    print("Got result:", result)

    print("\n--- Continue execution ---")
    suppress_and_continue()
