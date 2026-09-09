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

import std.os
from std.os.path import expandvars

from std.testing import TestSuite, assert_equal


@fieldwise_init
struct EnvVar(ImplicitlyCopyable):
    var name: String

    def __init__(out self, name: String, value: String):
        self.name = name
        _ = std.os.setenv(name, value)

    def __enter__(self) -> Self:
        return self

    def __exit__(self) -> None:
        _ = std.os.unsetenv(self.name)


def test_expansion() raises:
    with EnvVar("TEST_VAR", "World"):
        assert_equal(expandvars("Hello $TEST_VAR!"), "Hello World!")
        assert_equal(expandvars("漢字 $TEST_VAR🔥!"), "漢字 World🔥!")
        assert_equal(expandvars("$TEST_VAR/path/to/file"), "World/path/to/file")

    with EnvVar("UNICODE_TEST_VAR", "漢字🔥"):
        assert_equal(expandvars("Hello $UNICODE_TEST_VAR!"), "Hello 漢字🔥!")
        assert_equal(expandvars("漢字 $UNICODE_TEST_VAR🔥!"), "漢字 漢字🔥🔥!")
        assert_equal(
            expandvars("$UNICODE_TEST_VAR/path/to/file"), "漢字🔥/path/to/file"
        )


def test_braced_expansion() raises:
    with EnvVar("BRACE_VAR", "World"):
        assert_equal(expandvars("Hello ${BRACE_VAR}!"), "Hello World!")
        assert_equal(expandvars("漢字 ${BRACE_VAR}🔥!"), "漢字 World🔥!")
        assert_equal(
            expandvars("${BRACE_VAR}/path/to/file"), "World/path/to/file"
        )

    with EnvVar("UNICODE_BRACE_VAR", "漢字🔥"):
        assert_equal(expandvars("Hello ${UNICODE_BRACE_VAR}!"), "Hello 漢字🔥!")
        assert_equal(expandvars("漢字 ${UNICODE_BRACE_VAR}🔥!"), "漢字 漢字🔥🔥!")
        assert_equal(
            expandvars("${UNICODE_BRACE_VAR}/path/to/file"), "漢字🔥/path/to/file"
        )


def test_unset_expansion() raises:
    # Unset variables should not be expanded.
    assert_equal(
        expandvars("Hello $NONEXISTENT_VAR!"), "Hello $NONEXISTENT_VAR!"
    )
    assert_equal(
        expandvars("漢字 ${NONEXISTENT_VAR}🔥!"), "漢字 ${NONEXISTENT_VAR}🔥!"
    )


def test_dollar_sign() raises:
    # A lone `$` should not be expanded.
    assert_equal(expandvars("A lone $ sign"), "A lone $ sign")

    # Special shell variables should not be expanded.
    assert_equal(
        expandvars("$@ $* $1 $2 $3 $NONEXISTENT_VAR."),
        "$@ $* $1 $2 $3 $NONEXISTENT_VAR.",
    )


def test_short_variable() raises:
    with EnvVar("a", "World"):
        assert_equal(expandvars("$a"), "World")
        assert_equal(expandvars("${a}"), "World")


def test_invalid_syntax() raises:
    # Invalid syntax should be written as is.
    assert_equal(expandvars("${}"), "${}")
    assert_equal(expandvars("${"), "${")
    assert_equal(expandvars("${abc"), "${abc")
    assert_equal(expandvars("prefix ${abc"), "prefix ${abc")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
