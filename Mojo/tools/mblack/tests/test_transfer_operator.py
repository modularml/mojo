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

# Tests for the ownership transfer operator (^) with member access.

import pytest

from tests.util import assert_mojo_format

# A small struct with methods returning Int, used by every test snippet.
_STRUCT = (
    "struct Val:\n"
    "    var v: Int\n"
    "\n"
    "    def __init__(out self, x: Int = 0):\n"
    "        self.v = x\n"
    "\n"
    "    def finish(var self) -> Int:\n"
    "        return self.v\n"
    "\n"
    "    def get(var self) -> Int:\n"
    "        return self.v\n"
    "\n"
    "    def total(var self) -> Int:\n"
    "        return self.v\n"
    "\n"
    "    def hash(var self) -> Int:\n"
    "        return self.v\n"
    "\n"
    "    def a(var self) -> Val:\n"
    "        return Val(self.v)\n"
    "\n"
    "    def b(var self) -> Int:\n"
    "        return self.v\n"
    "\n"
    "\n"
)


def _wrap_val_struct(body: str) -> str:
    """Wrap a body in a main function preceded by the Val struct."""
    return _STRUCT + "def main():\n" + body


@pytest.mark.parametrize(
    "op",
    ["**", "*", "/", "%", "//", "+", "-", "<<", ">>", "&", "|", "^"],
)
def test_transfer_member_access_then_binop(op):
    """Transfer followed by member access and a binary operator."""
    source = _wrap_val_struct(f"    var x = Val()^.get() {op} 2\n")
    assert_mojo_format(source, source)


def test_transfer_member_access_then_multiply_in_parens():
    """Transfer with multiply in parenthesized shift expression."""
    source = _wrap_val_struct(
        "    var hasher = Val()\n"
        "    var factor = 2\n"
        "    var x = (hasher^.finish() * factor) >> 32\n"
    )
    assert_mojo_format(source, source)


def test_transfer_on_rhs_of_multiply():
    """Transfer on right side of *."""
    source = _wrap_val_struct(
        "    var hasher = Val()\n"
        "    var factor = 2\n"
        "    var x = factor * hasher^.finish()\n"
    )
    assert_mojo_format(source, source)


def test_transfer_standalone():
    """Plain transfer without member access."""
    source = _wrap_val_struct(
        "    var v: Int = 42\n"
        "    var x = v^\n"
    )
    assert_mojo_format(source, source)


def test_transfer_member_access_no_trailing_op():
    """Transfer with member access but no trailing binary op."""
    source = _wrap_val_struct("    var x = Val()^.finish()\n")
    assert_mojo_format(source, source)


def test_xor_still_works():
    """Plain XOR expressions must still parse and format."""
    source = _wrap_val_struct(
        "    var a = 5\n"
        "    var b = 3\n"
        "    var x = a ^ b\n"
    )
    assert_mojo_format(source, source)


def test_transfer_member_chain():
    """Transfer with chained member access then multiply."""
    source = _wrap_val_struct("    var x = Val()^.a().b() * 2\n")
    assert_mojo_format(source, source)


def test_transfer_with_trailing_comment():
    """Transfer expression with a trailing comment."""
    source = _wrap_val_struct(
        "    var hasher = Val()\n"
        "    var factor = 2\n"
        "    var x = hasher^.finish() * factor  # hash\n"
    )
    assert_mojo_format(source, source)


def test_transfer_power_then_multiply():
    """Transfer with ** followed by * (tests boundary in grammar)."""
    source = _wrap_val_struct("    var x = Val()^.get() ** 2 * 3\n")
    assert_mojo_format(source, source)


def test_transfer_then_xor():
    """Transfer followed by XOR in the same expression."""
    source = _wrap_val_struct(
        "    var mask = 0xFF\n"
        "    var x = Val()^.hash() ^ mask\n"
    )
    assert_mojo_format(source, source)


def test_transfer_space_before_dot():
    """Transfer with space between ^ and . gets normalized."""
    source = _wrap_val_struct("    var x = Val()^ .finish()\n")
    expected = _wrap_val_struct("    var x = Val()^.finish()\n")
    assert_mojo_format(source, expected)


def test_transfer_across_lines():
    """Transfer split across two lines gets joined."""
    source = _wrap_val_struct(
        "    var a = Val()\n"
        "    var x = a^\n"
        "        .finish()\n"
    )
    expected = _wrap_val_struct(
        "    var a = Val()\n"
        "    var x = a^.finish()\n"
    )
    assert_mojo_format(source, expected)


def test_transfer_across_lines_with_comment():
    """Transfer split across two lines with a trailing comment."""
    source = _wrap_val_struct(
        "    var a = Val()\n"
        "    var x = a^  # transfer\n"
        "        .finish()\n"
    )
    expected = _wrap_val_struct(
        "    var a = Val()\n"
        "    var x = a^.finish()  # transfer\n"
    )
    assert_mojo_format(source, expected)


def test_transfer_in_fmt_off():
    """Transfer inside a fmt-off region is preserved as-is."""
    source = _wrap_val_struct(
        "    # fmt: off\n"
        "    var x=Val()^.finish()*2\n"
        "    # fmt: on\n"
    )
    assert_mojo_format(source, source)


def test_transfer_complex_expression():
    """Transfer in a more complex arithmetic expression."""
    source = _wrap_val_struct(
        "    var b = 3\n"
        "    var x = (Val()^.get() * b + Val()^.get()) >> 8\n"
    )
    assert_mojo_format(source, source)
