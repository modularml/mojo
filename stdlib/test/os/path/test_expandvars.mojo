# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo %s

import os
from os.path import expandvars
from testing import assert_equal


def test_expansion():
    _ = os.setenv("TEST_VAR", "World")
    assert_equal(expandvars("Hello $TEST_VAR!"), "Hello World!")
    assert_equal(expandvars("漢字 $TEST_VAR🔥!"), "漢字 World🔥!")
    assert_equal(expandvars("$TEST_VAR/path/to/file"), "World/path/to/file")

    _ = os.setenv("UNICODE_TEST_VAR", "漢字🔥")
    assert_equal(expandvars("Hello $UNICODE_TEST_VAR!"), "Hello 漢字🔥!")
    assert_equal(expandvars("漢字 $UNICODE_TEST_VAR🔥!"), "漢字 漢字🔥🔥!")
    assert_equal(
        expandvars("$UNICODE_TEST_VAR/path/to/file"), "漢字🔥/path/to/file"
    )


def test_braced_expansion():
    _ = os.setenv("BRACE_VAR", "World")
    assert_equal(expandvars("Hello ${BRACE_VAR}!"), "Hello World!")
    assert_equal(expandvars("漢字 ${BRACE_VAR}🔥!"), "漢字 World🔥!")
    assert_equal(expandvars("${BRACE_VAR}/path/to/file"), "World/path/to/file")

    _ = os.setenv("UNICODE_BRACE_VAR", "漢字🔥")
    assert_equal(expandvars("Hello ${UNICODE_BRACE_VAR}!"), "Hello 漢字🔥!")
    assert_equal(expandvars("漢字 ${UNICODE_BRACE_VAR}🔥!"), "漢字 漢字🔥🔥!")
    assert_equal(
        expandvars("${UNICODE_BRACE_VAR}/path/to/file"), "漢字🔥/path/to/file"
    )


def test_unset_expansion():
    # Unset variables should be expanded to an empty string.
    assert_equal(
        expandvars("Hello $NONEXISTENT_VAR!"), "Hello $NONEXISTENT_VAR!"
    )
    assert_equal(
        expandvars("漢字 ${NONEXISTENT_VAR}🔥!"), "漢字 ${NONEXISTENT_VAR}🔥!"
    )


def test_dollar_sign():
    # A lone dollar sign should not be expanded.
    assert_equal(expandvars("A lone $ sign"), "A lone $ sign")

    # Special shell variables should not be expanded.
    assert_equal(
        expandvars("$@ $* $1 $2 $3 $NONEXISTENT_VAR."),
        "$@ $* $1 $2 $3 $NONEXISTENT_VAR.",
    )


def test_invalid_syntax():
    # Invalid syntax should be written as is.
    assert_equal(expandvars("${}"), "${}")
    assert_equal(expandvars("${"), "${")


def main():
    test_expansion()
    test_braced_expansion()
    test_unset_expansion()
    test_dollar_sign()
    test_invalid_syntax()
