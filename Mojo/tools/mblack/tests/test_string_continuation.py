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

# Tests for implicit string concatenation across continuation lines.

from tests.util import assert_mojo_format


def test_implicit_string_concat():
    """Adjacent string literals on continuation lines should be joined."""
    source = (
        "def main():\n"
        '    var long_text = "This is a long line of text that is easier to read if"\n'
        '                     " it is broken up across two lines."\n'
    )
    expected = (
        "def main():\n"
        "    var long_text = (\n"
        '        "This is a long line of text that is easier to read if"\n'
        '        " it is broken up across two lines."\n'
        "    )\n"
    )
    assert_mojo_format(source, expected)


def test_implicit_string_concat_with_backslash():
    """Backslash-continued string concatenation should also work."""
    source = (
        "def main():\n"
        '    var long_text = "first part" \\\n'
        '                     " second part"\n'
    )
    expected = (
        "def main():\n"
        '    var long_text = "first part second part"\n'
    )
    assert_mojo_format(source, expected)


def test_implicit_string_concat_single_quoted():
    """Single-quoted string continuation should also be joined."""
    source = (
        "def main():\n"
        "    var s = 'hello'\n"
        "             ' world'\n"
    )
    # Normalize single quotes to double quotes and merge adjacent strings.
    expected = (
        'def main():\n'
        '    var s = "hello world"\n'
    )
    assert_mojo_format(source, expected)


def test_implicit_string_concat_with_comment():
    """A trailing comment on the first string line should be preserved."""
    source = (
        "def main():\n"
        '    var s = "hello"  # greeting\n'
        '             " world"\n'
    )
    expected = (
        "def main():\n"
        '    var s = "hello world"  # greeting\n'
    )
    assert_mojo_format(source, expected)


def test_implicit_string_concat_chained():
    """Three consecutive string lines should all be joined."""
    source = (
        "def main():\n"
        '    var s = "one"\n'
        '             " two"\n'
        '             " three"\n'
    )
    expected = (
        "def main():\n"
        '    var s = "one two three"\n'
    )
    assert_mojo_format(source, expected)


def test_string_deeper_indent_after_colon_not_joined():
    """A deeper-indented string after a colon line should not be joined."""
    source = (
        "def main():\n"
        "    if True:\n"
        '        "hello"\n'
    )
    assert_mojo_format(source, source)


def test_implicit_string_concat_inside_triple_quoted_not_joined():
    """Lines inside a triple-quoted string should not be joined."""
    source = (
        'def main():\n'
        '    var s = """\n'
        '    "hello"\n'
        '    " world"\n'
        '    """\n'
    )
    assert_mojo_format(source, source)


def test_docstring_after_blank_line_not_joined():
    """A docstring after a blank line inside a struct should not be joined."""
    source = (
        "struct Foo:\n"
        "\n"
        '    """Docstring."""\n'
        "\n"
        "    var x: Int\n"
    )
    assert_mojo_format(source, source)


def test_standalone_string_after_non_string_line_not_joined():
    """A string literal after a non-string statement should not be joined."""
    source = (
        "def main():\n"
        "    var x = 1\n"
        '    """Block comment."""\n'
    )
    assert_mojo_format(source, source)


def test_implicit_string_concat_fmt_off():
    """String concatenation inside a fmt: off block should not be joined.

    Tests ``_normalize_mojo_source`` directly because the un-normalized
    syntax crashes the lib2to3 parser.
    """
    from mblib2to3.pgen2.driver import _normalize_mojo_source

    source = (
        "def main():\n"
        "    # fmt: off\n"
        '    var s = "hello"\n'
        '             " world"\n'
        "    # fmt: on\n"
        "    print(s)\n"
    )
    assert _normalize_mojo_source(source) == source


def test_trailing_operator_then_string_not_duplicated():
    """A string on a continuation line after a trailing operator should be
    joined once."""
    source = (
        "def main():\n"
        '    var a = "hello"\n'
        "    var result = a +\n"
        '        " world"\n'
    )
    expected = (
        "def main():\n"
        '    var a = "hello"\n'
        '    var result = a + " world"\n'
    )
    assert_mojo_format(source, expected)


def test_trailing_operator_then_ternary_not_duplicated():
    """Ternary continuation after a trailing operator should be joined once."""
    source = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var cond = True\n"
        "    var c = 3\n"
        "    var x = a +\n"
        "        b if cond else c\n"
    )
    expected = (
        "def main():\n"
        "    var a = 1\n"
        "    var b = 2\n"
        "    var cond = True\n"
        "    var c = 3\n"
        "    var x = a + b if cond else c\n"
    )
    assert_mojo_format(source, expected)
