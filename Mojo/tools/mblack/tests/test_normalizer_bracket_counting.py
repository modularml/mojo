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

# Tests that the Mojo line-continuation normalizer counts only brackets that
# appear in *code*.  An unbalanced bracket inside a comment, a string literal,
# or a docstring must not desync the normalizer's bracket-depth tracking and
# silently disable line rejoining for the rest of the scope.

from tests.util import assert_mojo_format


def test_unmatched_paren_in_comment_does_not_disable_continuation():
    """A "(" in a comment must not stop a later continuation from rejoining."""
    source = (
        "def main():\n"
        "    # returns a (partial result\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    # returns a (partial result\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_string_does_not_disable_continuation():
    """A "(" inside a string literal must not stop a later continuation --
    including after an escaped quote, which must not end the string early."""
    source = (
        "def main():\n"
        '    var s = "an unmatched ( paren in a string"\n'
        '    var s2 = "a \\" b ("\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    var s = "an unmatched ( paren in a string"\n'
        "    var s2 = 'a \" b ('\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_oneline_docstring_does_not_disable_continuation():
    """A "(" in a one-line docstring must not desync bracket tracking."""
    source = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren."""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren."""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_multiline_docstring_does_not_disable_continuation():
    """A "(" on a multi-line docstring's summary whose matching ")" lives in the
    body. Also covers a string closing mid-line before trailing code, and an
    f-string interpolation spanning the line break. Both must still net zero
    depth so the continuation rejoins."""
    source = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren.\n'
        "\n"
        "    More body text (with a balanced pair) here.\n"
        '    """\n'
        "    var a = 1\n"
        "    var b = 2\n"
        '    var s = """multi\n'
        '    line""".upper()\n'
        '    var f = t"""x {a +\n'
        '        b} y"""\n'
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    """Summary with one ( unbalanced paren.\n'
        "\n"
        "    More body text (with a balanced pair) here.\n"
        '    """\n'
        "    var a = 1\n"
        "    var b = 2\n"
        '    var s = """multi\n'
        '    line""".upper()\n'
        '    var f = t"""x {a +\n'
        '        b} y"""\n'
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_tstring_interp_does_not_disable_continuation():
    """A nested string in a t-string interpolation that reuses the outer quote
    must not be read as the terminator, leaking its "(" as code. Also exercises
    other interpolation shapes -- escaped braces, a dict/list body, a
    different-quote nested string, and a nested t-string -- all of which are
    complete literals that must net zero bracket depth."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        '    var s = t"{"("}"\n'
        '    var s2 = t"{{literal}}"\n'
        '    var s3 = t"{ {a: a} }"\n'
        '    var s4 = t"{t"{a}"}"\n'
        "    var s5 = t\"{'y'}\"\n"
        '    var s6 = t"{[a, a]}"\n'
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        '    var s = t"{"("}"\n'
        '    var s2 = t"{{literal}}"\n'
        '    var s3 = t"{ {a: a} }"\n'
        '    var s4 = t"{t"{a}"}"\n'
        "    var s5 = t\"{'y'}\"\n"
        '    var s6 = t"{[a, a]}"\n'
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_unmatched_paren_in_singlequote_tstring_interp_does_not_disable():
    """The same nested-quote hazard, but with single-quoted strings."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var s = t'{'('}'\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var s = t'{'('}'\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_real_open_bracket_after_tstring_is_still_tracked():
    """Guard against over-correcting: a complete t-string nets zero depth, so
    a real open "[" on the next line is still tracked as a continuation."""
    source = (
        "def main():\n"
        '    var s = t"{"("}"\n'
        "    var x = [\n"
        "        1 +\n"
        "        2,\n"
        "    ]\n"
    )
    expected = (
        "def main():\n"
        '    var s = t"{"("}"\n'
        "    var x = [\n"
        "        1 + 2,\n"
        "    ]\n"
    )
    assert_mojo_format(source, expected)


def test_oneline_triple_string_closing_at_eol_does_not_carry_over():
    """A triple-quoted string whose closing quote ends the line must not be
    mistaken for an unterminated one, which would disable the rejoin below."""
    source = (
        "def main():\n"
        '    var s = """one line triple"""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        '    var s = """one line triple"""\n'
        "    var a = 1\n"
        "    var b = 2\n"
        "    var x = a + b\n"
    )
    assert_mojo_format(source, expected)


def test_escaped_quote_keeps_triple_string_open_across_lines():
    """An escaped quote inside a triple-quoted string keeps the string open
    until a later closing delimiter; the scanner must not close it early (at
    the escaped quote) and disable the continuation below."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        '    var s = """abc\\"""\n'
        'xyz"""\n'
        "    var x = a +\n"
        "        a\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        '    var s = """abc\\"""\n'
        'xyz"""\n'
        "    var x = a + a\n"
    )
    assert_mojo_format(source, expected)


def test_keyword_flush_against_string_is_not_read_as_interp_prefix():
    """A keyword flush against a string (``if"{"``) must not be read as an
    f/t-string prefix. If it is, the phantom interpolation eats the closing
    quote, the ")" is dropped, and depth stays stuck so the ``a +`` below
    never rejoins."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        '    var x = (a if"{" else b)\n'
        "    var y = a +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        '    var x = a if "{" else b\n'
        "    var y = a + b\n"
    )
    assert_mojo_format(source, expected)
