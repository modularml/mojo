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
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from typed_errors_edge_cases import (
    NetworkError,
    ParseError,
    ApplicationError,
    fetch_data,
    parse_data,
    fetch_and_parse,
    level1,
    level2,
    level3,
    reraise_with_logging,
    reraise_as_different_type,
    reraise_modified,
    suppress_error,
    suppress_and_continue,
)


def test_network_error_fields() raises:
    """Test NetworkError field access."""
    try:
        _ = fetch_data()
    except e:
        assert_equal(e.code, 404)
        assert_equal(e.message, "Resource not found")


def test_parse_error_fields() raises:
    """Test ParseError field access."""
    try:
        _ = parse_data("test")
    except e:
        assert_equal(e.position, 0)
        assert_equal(e.expected, "integer")


def test_nested_try_blocks() raises:
    """Test nested try blocks with different error types."""
    try:
        _ = fetch_and_parse()
    except e:
        assert_equal(e.source, "network")
        assert_true(e.details.find("NetworkError") >= 0)


def test_error_propagation_level3() raises:
    """Test error originates from level3."""
    with assert_raises(contains="Server error"):
        level3()


def test_error_propagation_level2() raises:
    """Test error propagates through level2."""
    with assert_raises(contains="Server error"):
        level2()


def test_error_propagation_level1() raises:
    """Test error propagates through entire chain."""
    try:
        level1()
    except e:
        assert_equal(e.code, 500)
        assert_equal(e.message, "Server error")


def test_reraise_with_logging() raises:
    """Test re-raising preserves error."""
    try:
        _ = reraise_with_logging()
    except e:
        assert_equal(e.code, 404)
        assert_equal(e.message, "Resource not found")


def test_reraise_as_different_type() raises:
    """Test converting error type on re-raise."""
    try:
        _ = reraise_as_different_type()
    except e:
        assert_equal(e.source, "network")
        assert_true(e.details.find("404") >= 0)


def test_reraise_modified() raises:
    """Test modifying error on re-raise."""
    try:
        _ = reraise_modified()
    except e:
        assert_equal(e.code, 404)
        assert_true(e.message.find("[Modified]") >= 0)


def test_suppress_error_returns_default() raises:
    """Test error suppression returns default value."""
    var result = suppress_error()
    assert_equal(result, "default value")


def test_suppress_and_continue() raises:
    """Test error suppression allows continuation."""
    # This should complete without raising
    suppress_and_continue()


def test_application_error_from_network() raises:
    """Test ApplicationError factory from NetworkError."""
    var net_err = NetworkError(code=500, message="test")
    var app_err = ApplicationError.from_network(net_err)
    assert_equal(app_err.source, "network")
    assert_true(app_err.details.find("500") >= 0)


def test_application_error_from_parse() raises:
    """Test ApplicationError factory from ParseError."""
    var parse_err = ParseError(position=10, expected="number")
    var app_err = ApplicationError.from_parse(parse_err)
    assert_equal(app_err.source, "parse")
    assert_true(app_err.details.find("10") >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
