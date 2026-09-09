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

# Tests for multi-line ternary expression formatting.

import pytest
import mblack.parsing
from tests.util import assert_mojo_format, mojo_format_str


def test_multiline_ternary_var_assignment():
    """mblack should format a var assignment with a multi-line ternary."""
    source = (
        "def main():\n"
        "    var b = 7\n"
        "    var a = 10\n"
        "       if b % 2 else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var b = 7\n"
        "    var a = 10 if b % 2 else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_already_single_line():
    """A ternary already on one line should be left unchanged."""
    source = (
        "def main():\n"
        "    var b = 7\n"
        "    var a = 10 if b % 2 else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, source)


def test_multiline_ternary_return():
    """mblack should format a return statement with a multi-line ternary."""
    source = (
        "def foo(b: Int) -> Int:\n"
        "    return 10\n"
        "        if b % 2 else 100\n"
    )
    expected = (
        "def foo(b: Int) -> Int:\n"
        "    return 10 if b % 2 else 100\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_nested():
    """mblack should handle nested ternary expressions split across lines."""
    source = (
        "def foo(x: Int) -> Int:\n"
        "    return 1\n"
        "        if x > 0 else -1\n"
        "        if x < 0 else 0\n"
    )
    expected = (
        "def foo(x: Int) -> Int:\n"
        "    return 1 if x > 0 else -1 if x < 0 else 0\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_nested_fully_split():
    """Chained ternary where every if and else is on its own line."""
    source = (
        "def foo(x: Int):\n"
        "    var a = 1\n"
        "        if x > 0\n"
        "        else -1\n"
        "            if x < 0\n"
        "            else 0\n"
        "    print(a)\n"
    )
    expected = (
        "def foo(x: Int):\n"
        "    var a = 1 if x > 0 else -1 if x < 0 else 0\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_if_no_space_before_paren():
    """``if(cond)`` without a space should still be recognized as a
    ternary continuation."""
    source = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10\n"
        "        if(b % 2) else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10 if (b % 2) else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_identifier_starting_with_keyword_not_joined():
    """Identifiers like ``ifvar`` or ``elsevar`` must not be mistaken for
    ternary ``if``/``else`` keywords."""
    source = (
        "def main():\n"
        "    var x = 10\n"
        "    ifvar = 20\n"
        "    elsevar = 30\n"
        "    print(x)\n"
    )
    assert_mojo_format(source, source)


def test_indented_if_statement_not_joined():
    """An indented ``if`` statement (with colon) must not be treated as a
    ternary continuation — the parser should still report an indent error."""
    source = (
        "def main():\n"
        "  x = 0\n"
        "   if x > 0:\n"
        "    x = 2\n"
    )
    with pytest.raises(mblack.parsing.InvalidInput, match="Unexpected indent"):
        mojo_format_str(source)


def test_multiline_else_on_separate_line():
    """``else`` on its own further-indented line is valid Mojo and should
    be joined back to the previous line."""
    source = (
        "def main():\n"
        "    var a = 10 if True\n"
        "        else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var a = 10 if True else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_if_and_else_on_separate_lines():
    """Both ``if`` and ``else`` on their own continuation lines."""
    source = (
        "def main():\n"
        "    var a = 10\n"
        "        if True\n"
        "        else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var a = 10 if True else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_inside_triple_quoted_string():
    """Lines inside a triple-quoted string should not be joined."""
    source = (
        'def main():\n'
        '    var s = """\n'
        '    10\n'
        '       if b % 2 else 100\n'
        '    """\n'
        '    print(s)\n'
    )
    assert_mojo_format(source, source)


@pytest.mark.parametrize("spaces", ["", " ", "  ", "   ", "\t"])
def test_multiline_ternary_with_comment_on_prev_line(spaces):
    """A trailing comment on the previous line must not swallow the
    ternary continuation, regardless of spacing before #."""
    source = (
        "def main():\n"
        "    var b = 0\n"
        f"    var a = 10{spaces}# pick ten\n"
        "        if b % 2 else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10 if b % 2 else 100  # pick ten\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_with_comment_on_continuation_line():
    """A trailing comment on the continuation line should be preserved."""
    source = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10\n"
        "        if b % 2 else 100  # fallback\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10 if b % 2 else 100  # fallback\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_with_comments_on_both_lines():
    """When both lines carry comments, they are concatenated — matching
    how mblack collapses parenthesized arithmetic with comments::

        var a = (
            1  # first
            + 2  # second
        )
        ->  var a = 1 + 2  # first  # second
    """
    source = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10  # value\n"
        "        if b % 2 else 100  # fallback\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10 if b % 2 else 100  # fallback  # value\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_with_hash_in_string_and_comment():
    """A # inside a string literal must not be confused with a comment."""
    source = (
        "def foo(x: String) -> Int:\n"
        "    return 0\n"
        "\n"
        "\n"
        "def main():\n"
        "    var bar = 0\n"
        '    var a = foo("a#b")  # note\n'
        "        if True else bar\n"
        "    print(a)\n"
    )
    expected = (
        "def foo(x: String) -> Int:\n"
        "    return 0\n"
        "\n"
        "\n"
        "def main():\n"
        "    var bar = 0\n"
        '    var a = foo("a#b") if True else bar  # note\n'
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_backslash_continuations_get_joined():
    """A ternary with backslash continuations should be collapsed to one line."""
    source = (
        "def main():\n"
        "    var a = 10 \\\n"
        "        if True \\\n"
        "        else 100\n"
        "    print(a)\n"
    )
    expected = (
        "def main():\n"
        "    var a = 10 if True else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_tab_indented():
    """Ternary continuation with tab indentation should be joined correctly."""
    source = (
        "def main():\n"
        "\tvar b = 0\n"
        "\tvar a = 10\n"
        "\t\tif b % 2 else 100\n"
        "\tprint(a)\n"
    )
    # mblack normalizes tabs to spaces in its output.
    expected = (
        "def main():\n"
        "    var b = 0\n"
        "    var a = 10 if b % 2 else 100\n"
        "    print(a)\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_ternary_fmt_off():
    """A multi-line ternary inside a fmt: off block should not be joined.

    Tests ``_normalize_mojo_source`` directly because the un-normalized
    multi-line ternary crashes the lib2to3 parser.  If the parser learns
    to handle this syntax natively, this test can be changed to use
    ``assert_mojo_format`` instead.
    """
    from mblib2to3.pgen2.driver import _normalize_mojo_source

    source = (
        "def main():\n"
        "    # fmt: off\n"
        "    var a = 10\n"
        "        if True else 100\n"
        "    # fmt: on\n"
        "    print(a)\n"
    )
    assert _normalize_mojo_source(source) == source


def test_multiline_if_statement_after_comment_not_joined():
    """A multi-line ``if`` statement after a line with a trailing comment
    must not be mistaken for a ternary continuation."""
    source = (
        "def main():\n"
        "    var n = 0\n"
        "    var limit = 0\n"
        "    for i in range(0, n, 128):  # stride\n"
        "        if (\n"
        "            i < limit\n"
        "        ):\n"
        "            print(i)\n"
    )
    expected = (
        "def main():\n"
        "    var n = 0\n"
        "    var limit = 0\n"
        "    for i in range(0, n, 128):  # stride\n"
        "        if i < limit:\n"
        "            print(i)\n"
    )
    assert_mojo_format(source, expected)


def test_ternary_like_inside_brackets_not_joined():
    """Lines inside open brackets should not be joined by the normalizer.

    The tokenizer already handles implicit line continuation inside
    brackets, so the normalizer must not interfere.
    """
    from mblib2to3.pgen2.driver import _normalize_mojo_source

    source = (
        "def foo(v: Int) -> Int:\n"
        "    return v\n"
        "\n"
        "\n"
        "def main():\n"
        "    var cond = True\n"
        "    var value = 0\n"
        "    var other = 0\n"
        "    var x = foo(\n"
        "        value\n"
        "        if cond else other\n"
        "    )\n"
    )
    # The normalizer should leave the source unchanged — the tokenizer
    # handles implicit line continuation inside brackets.
    assert _normalize_mojo_source(source) == source
    expected = (
        "def foo(v: Int) -> Int:\n"
        "    return v\n"
        "\n"
        "\n"
        "def main():\n"
        "    var cond = True\n"
        "    var value = 0\n"
        "    var other = 0\n"
        "    var x = foo(value if cond else other)\n"
    )
    assert_mojo_format(source, expected)
