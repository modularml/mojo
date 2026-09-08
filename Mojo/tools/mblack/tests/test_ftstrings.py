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

"""Tests for f-strings and t-strings with nested quotes (same quote character)."""

import pytest
from tests.util import assert_mojo_format, mojo_format_str


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_double_quote_nested(prefix):
    """Test nested double quotes in double-quoted string."""
    source = f'{prefix}"hello {{"world"}}"'
    expected = f'{prefix}"hello {{"world"}}"\n'
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_single_quote_nested(prefix):
    """Test nested single quotes in single-quoted string."""
    source = f"{prefix}'hello {{'world'}}'"  # noqa: F541
    expected = f"{prefix}'hello {{'world'}}'\n"
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_multiple_interpolations(prefix):
    """Test multiple nested interpolations."""
    source = f'{prefix}"a {{"b"}} c {{"d"}}"'
    expected = f'{prefix}"a {{"b"}} c {{"d"}}"\n'
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_triple_quoted(prefix):
    """Test triple-quoted strings with nested quotes."""
    source = f'{prefix}"""hello {{"world"}}"""'
    expected = f'{prefix}"""hello {{"world"}}"""\n'
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_complex_expression(prefix):
    """Test complex expression with function calls."""
    source = f'{prefix}"result: {{foo("bar", "baz")}}"'
    expected = f'{prefix}"result: {{foo("bar", "baz")}}"\n'
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_escaped_braces(prefix):
    """Test escaped braces alongside interpolations."""
    source = prefix + '"data: {{key: {"value"}}}"'
    expected = prefix + '"data: {{key: {"value"}}}"\n'
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_multiline(prefix):
    """Test multiline triple-quoted strings."""
    source = f'''{prefix}"""
hello {{"world"}}
and {{"universe"}}
"""'''
    result = mojo_format_str(source)
    assert "hello" in result and "world" in result


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_nested_different_quotes(prefix):
    """Test nested strings using different quote characters."""
    source = f'''{prefix}"hello {{'world'}}"'''
    expected = f'''{prefix}"hello {{'world'}}"\n'''
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_deeply_nested(prefix):
    """Test deeply nested same-quote strings."""
    source = prefix + '"a {' + prefix + '"b {' + prefix + '"c"}"}"'
    result = mojo_format_str(source)
    assert "a" in result and "b" in result and "c" in result


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_malformed_input_raises(prefix):
    """Test that malformed input raises an exception."""
    source = f'{prefix}"hello {{world"'  # Unclosed brace
    with pytest.raises(Exception):
        mojo_format_str(source)


def test_fstring_quote_normalization_skipped():
    """F-strings skip quote normalization to preserve interpolations."""
    source = r'f"test {"say \"hello\""}"'
    expected = 'f"test {"say \\"hello\\""}"\n'
    assert_mojo_format(source, expected)


def test_mixed_fstring_and_tstring():
    """Test f-strings and t-strings can coexist."""
    source = 'x = f"hello {name}" + t"world {value}"'
    result = mojo_format_str(source)
    assert "f" in result and "t" in result


@pytest.mark.parametrize(
    "prefix",
    ["rt", "rT", "Rt", "RT", "tr", "tR", "Tr", "TR"],
)
def test_raw_tstring_prefix_normalized_to_rt(prefix):
    """All raw t-string prefix variants are normalized to 'rt'."""
    source = f'{prefix}"hello {{name}}"'
    assert_mojo_format(source, 'rt"hello {name}"\n')


@pytest.mark.parametrize(
    "prefix",
    ["rf", "rF", "Rf", "RF", "fr", "fR", "Fr", "FR"],
)
def test_raw_fstring_prefix_normalized_to_rf(prefix):
    """All raw f-string prefix variants are normalized to 'rf'."""
    source = f'{prefix}"hello {{name}}"'
    assert_mojo_format(source, 'rf"hello {name}"\n')


def _assert_all_parts_have_t_prefix(result: str) -> None:
    """Check that every string token in result starts with 't'."""
    import re

    string_tokens = re.findall(r'[tTfFrRbBuU]*"[^"]*"', result)
    assert len(string_tokens) > 1, (
        "Expected the t-string to be split into multiple parts"
    )
    for tok in string_tokens:
        assert tok.lower().startswith("t"), (
            f"String part {tok!r} lost its 't' prefix after splitting"
        )


def test_tstring_split_preserves_t_prefix_on_plain_part():
    """When a long t-string is split, parts WITHOUT interpolation keep 't'.

    This is the key difference from f-strings: the formatter may drop 'f'
    from expression-free parts of a split f-string, but must never drop 't'
    from any part of a split t-string because every part must remain a
    Template object.
    """
    # The interpolation {value} is near the end, so the first split part
    # will be plain text with no expressions — it must still start with t".
    source = (
        'x = t"this is a very long plain text segment without any interpolation'
        " at all and it just keeps going and then here comes"
        ' {value} at the end"\n'
    )
    result = mojo_format_str(source)
    _assert_all_parts_have_t_prefix(result)


def test_tstring_var_decl_splits():
    """A long t-string in a var declaration should be split across lines."""
    source = (
        'var x = t"this is a very long plain text segment without any'
        " interpolation at all and it just keeps going and then here"
        ' comes {value} at the end"\n'
    )
    result = mojo_format_str(source)
    _assert_all_parts_have_t_prefix(result)


def test_var_decl_plain_string_splits():
    """A long plain string in a var declaration should be split across lines."""
    source = (
        'var x = "this is a very long plain text segment that keeps going and'
        " going and going until it is way too long to fit on a single line"
        ' without wrapping"\n'
    )
    result = mojo_format_str(source)
    # Should be split across multiple lines
    assert result.strip().count("\n") >= 1, (
        "Expected the string in var declaration to be split across lines"
    )


def test_tstring_comptime_decl_splits():
    """A long t-string in a comptime declaration should be split, preserving t prefix."""
    source = (
        'comptime x = t"this is a very long plain text segment without any'
        " interpolation at all and it just keeps going and then here"
        ' comes {value} at the end"\n'
    )
    result = mojo_format_str(source)
    _assert_all_parts_have_t_prefix(result)


def test_comptime_decl_plain_string_splits():
    """A long plain string in a comptime declaration should be split across lines."""
    source = (
        'comptime x = "this is a very long plain text segment that keeps going and'
        " going and going until it is way too long to fit on a single line"
        ' without wrapping"\n'
    )
    result = mojo_format_str(source)
    # Should be split across multiple lines
    assert result.strip().count("\n") >= 1, (
        "Expected the string in comptime declaration to be split across lines"
    )


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_long_fstring_with_operator_in_interpolation_not_wrapped(prefix):
    """Long f/t-strings with binary operators inside {...} should not be wrapped.

    Regression test: spaces inside {expr // divisor} are not splittable by
    StringSplitter, so StringParenWrapper must not wrap the string even though
    the token value contains spaces (from around the // operator).
    """
    # This string exceeds 88 chars and contains spaces inside {num // group}.
    # It should be left as-is, not wrapped in (t"...").
    source = (
        "@__name(\n"
        f'    {prefix}"mha_depth{{config.depth}}_{{q_type}}_{{output_type}}_{{ragged}}_{{is_shared_kv}}_nqh{{config.num_heads}}_nkvh{{config.num_heads // group}}"\n'
        ")\n"
        "def kernel():\n"
        "    pass\n"
    )
    result = mojo_format_str(source)
    assert result == source, (
        f"Long {prefix}-string with // in interpolation should not be wrapped in parens"
    )


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_hash_in_interpolation_is_not_read_as_comment(prefix):
    """A "#" nested in an f/t-string interpolation (``t"{"#"}"``) must not be
    read as a trailing comment. If it is the ``.upper()`` continuation joins
    onto a truncated ``String(t"{"`` and corrupts the line."""
    source = (
        "def main():\n"
        f'    var s = String({prefix}"{{"#"}}")\n'
        "        .upper()\n"
    )
    expected = (
        "def main():\n"
        f'    var s = String({prefix}"{{"#"}}").upper()\n'
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_hash_in_triple_quoted_interpolation_is_not_read_as_comment(prefix):
    """The triple-quoted variant of the same hazard: a "#" nested in a
    triple-quoted f/t-string's interpolation must not be read as a comment."""
    source = (
        "def main():\n"
        f'    var s = String({prefix}"""{{"#"}}""")\n'
        "        .upper()\n"
    )
    expected = (
        "def main():\n"
        f'    var s = String({prefix}"""{{"#"}}""").upper()\n'
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize(
    "literal",
    [
        '"{{x}}"',  # escaped braces
        '"{ {1: 2} }"',  # nested braces: a dict literal in the interpolation
        '"{a:>{w}}"',  # nested braces in a format spec
    ],
)
@pytest.mark.parametrize("prefix", ["f", "t"])
def test_comment_after_braces_in_ftstring_is_stripped_before_merge(
    prefix, literal
):
    """A trailing comment after an f/t-string with escaped or nested braces
    must still be stripped before a continuation merges; otherwise the
    ``.upper()`` lands after the ``#`` and is silently commented out."""
    source = (
        "def main():\n"
        f"    var s = {prefix}{literal}  # comment\n"
        "        .upper()\n"
    )
    expected = (
        "def main():\n" + f"    var s = {prefix}{literal}.upper()  # comment\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("prefix", ["f", "t"])
def test_hash_in_interpolation_with_real_trailing_comment(prefix):
    """Both halves of the hazard at once: the "#" inside the interpolation
    must be skipped, while the real comment after the string is still
    stripped before the continuation merges (and reattached after)."""
    source = (
        "def main():\n"
        f'    var s = String({prefix}"{{"#"}}")  # real comment\n'
        "        .upper()\n"
    )
    expected = (
        "def main():\n"
        f'    var s = String({prefix}"{{"#"}}").upper()  # real comment\n'
    )
    assert_mojo_format(source, expected)
