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

# Tests for implicit line continuation with trailing operators and assignments.

import pytest
from tests.util import assert_mojo_format

# Note: `**` is excluded because we collapse spaces around power operations.
MOJO_BINARY_OPS = [
    "and", "or",
    "+", "-", "*", "/", "//", "%",
    "&", "|", "^", "<<", ">>",
    "==", "!=", "<", ">", "<=", ">=",
]

MOJO_ASSIGNMENT_OPS = ["=", "+=", "-=", "*=", "/=", "//=", "**=", "<<=", ">>="]


@pytest.mark.parametrize("op", MOJO_BINARY_OPS)
def test_trailing_binary_operator(op):
    """Implicit line continuation with a trailing binary operator should parse."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op}\n"
        "            b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op} b\n"
    )
    assert_mojo_format(source, expected)


def test_trailing_power_operator():
    """Trailing ``**`` joins and collapses spaces."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a **\n"
        "            b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a**b\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("op", MOJO_BINARY_OPS)
def test_trailing_operator_chain(op):
    """Multiple chained trailing operators across several lines."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var c = 3\n"
        f"    var x = a {op}\n"
        f"            b {op}\n"
        "            c\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var c = 3\n"
        f"    var x = a {op} b {op} c\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("op", MOJO_ASSIGNMENT_OPS)
def test_trailing_assignment(op):
    """Implicit line continuation with a trailing assignment operator should parse."""
    source = (
        "def main():\n"
        "    var x = 1\n"
        f"    x {op}\n"
        "        2\n"
    )
    expected = (
        "def main():\n"
        "    var x = 1\n"
        f"    x {op} 2\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("op", MOJO_BINARY_OPS)
def test_trailing_operator_with_comment(op):
    """Trailing comment on the operator line should be preserved."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op}  # comment\n"
        "            b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op} b  # comment\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("op", MOJO_BINARY_OPS)
def test_explicit_backslash_continuation(op):
    r"""Explicit ``\`` line continuation should still format correctly."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op} \\\n"
        "            b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        f"    var x = a {op} b\n"
    )
    assert_mojo_format(source, expected)


def test_ticket_repro_trailing_and():
    """Original reproduction case: trailing `and`."""
    source = (
        "def main():\n"
        "    var a = True\n"
        "    var b = True\n"
        "    var c = True\n"
        "    var d = True\n"
        "    var result = a == b and\n"
        "                 c == d\n"
    )
    expected = (
        "def main():\n"
        "    var a = True\n"
        "    var b = True\n"
        "    var c = True\n"
        "    var d = True\n"
        "    var result = a == b and c == d\n"
    )
    assert_mojo_format(source, expected)


def test_ticket_repro_trailing_equals():
    """Original reproduction case: trailing `=`."""
    source = (
        "def main():\n"
        "    var x = List[Int](length=4, fill=0)\n"
        "    x[0] =\n"
        "        x[0] + 1\n"
    )
    expected = (
        "def main():\n"
        "    var x = List[Int](length=4, fill=0)\n"
        "    x[0] = x[0] + 1\n"
    )
    assert_mojo_format(source, expected)
