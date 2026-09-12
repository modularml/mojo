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

"""Tests for the `where <condition> else "<message>"` form.

This is the alternative spelling of the message that `test_where_clause_message`
covers as `where (condition, "message")`. Unlike that one it needs a grammar
rule of its own -- `where_clause: WHERE top_test ['else' test]` -- because the
message sits outside any brackets.

The trailing `else` does not collide with a conditional expression's: that one
is only reachable after an `if`, so `if_else_test` has already consumed its own
`else` by the time the clause's is reached.
"""

from tests.util import assert_mojo_format

# ============================ #
# Fns with else messages
# ============================ #


def test_fn_where_clause_else_message():
    source = (
        "def gated[sc: Int]() where sc > 1 else"
        ' "scaling factor must be greater than 1": pass'
    )
    expected = (
        "def gated[sc: Int]() where sc > 1 else"
        ' "scaling factor must be greater than 1":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_short_where_clause_else_message():
    source = 'def gated[x: Bool]() where x else "x must hold": pass'
    expected = 'def gated[x: Bool]() where x else "x must hold":\n    pass\n'
    assert_mojo_format(source, expected)


def test_multiple_fn_where_clauses_with_else_messages():
    source = (
        "def multi[a: Int, b: Int]() "
        'where a > 0 else "a positive" where b > 0 else "b positive": pass'
    )
    expected = (
        "def multi[\n"
        "    a: Int, b: Int\n"
        ']() where a > 0 else "a positive" where b > 0 else "b positive":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_where_clause_message_forms_mix():
    """The two spellings are per-clause, so one signature may use both."""
    source = (
        "def mixed[a: Int, b: Int]() "
        'where (a > 0, "paren form") where b > 0 else "else form": pass'
    )
    expected = (
        "def mixed[\n"
        "    a: Int, b: Int\n"
        ']() where (a > 0, "paren form") where b > 0 else "else form":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_else_message_adjacent_literals():
    """Adjacent string literals in the message are merged when they fit on one
    line, as they are in the parenthesized form."""
    source = 'def concat[x: Int]() where x > 0 else "part one " "part two": pass'
    expected = (
        'def concat[x: Int]() where x > 0 else "part one part two":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_else_message_parenthesized():
    """The parser accepts parentheses around the message, which is how an
    author wraps a long one. Once the literals merge back to something that
    fits, the message collapses onto a single line inside its parentheses."""
    source = (
        "def wrapped[w: Int]() where w > 0 else (\n"
        '    "the width must be positive "\n'
        '    "and no wider than a warp"\n'
        "):\n"
        "    pass\n"
    )
    expected = (
        "def wrapped[\n"
        "    w: Int\n"
        ']() where w > 0 else ("the width must be positive and no wider than'
        ' a warp"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_else_message_prewrapped_too_long():
    """Author-supplied parentheses and split points are preserved when the
    message is too long to fit on one line. Unlike the parenthesized form,
    the message here sits outside any bracket, so the formatter cannot
    introduce the split itself -- see the module docstring."""
    source = (
        "def gated[n: Int]() where n > 0 else (\n"
        '    "the scaling factor must be greater than one for this"\n'
        '    " operation to be well defined"\n'
        "):\n"
        "    pass\n"
    )
    expected = (
        "def gated[\n"
        "    n: Int\n"
        "]() where n > 0 else (\n"
        '    "the scaling factor must be greater than one for this"\n'
        '    " operation to be well defined"\n'
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_else_message_ternary_condition():
    """A conditional expression owns the first `else`; the clause's message is
    introduced by the second."""
    source = (
        "def tern[A: Int, B: Int, C: Int]() "
        'where (C <= B) if (A <= B) else (C <= 0) else "bounded": pass'
    )
    expected = (
        "def tern[\n"
        "    A: Int, B: Int, C: Int\n"
        ']() where (C <= B) if (A <= B) else (C <= 0) else "bounded":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================ #
# Structs with else messages
# ================================ #


def test_struct_where_clause_else_message():
    source = 'struct Gated[N: Int] where N > 0 else "N must be positive": pass'
    expected = (
        'struct Gated[N: Int] where N > 0 else "N must be positive":\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ========================================== #
# Conformance lists with else messages
# ========================================== #


def test_conditional_conformance_else_message():
    source = (
        "struct Wrapped[T: Movable](\n"
        "    Copyable where conforms_to(\n"
        "        T, Copyable\n"
        '    ) else "wrapped type must be copyable",\n'
        "):\n"
        "    pass\n"
    )
    expected = source
    assert_mojo_format(source, expected)


def test_conditional_conformance_else_message_sorts_with_trait():
    """The conformance-list sorter reorders traits alphabetically. The message
    is part of its trait's `where` clause, so it moves with the trait and
    stays correctly attached -- the entry-separating comma is never mistaken
    for part of the clause, because `else` is a keyword."""
    source = (
        "struct Ordered[T: Movable](\n"
        "    Movable,\n"
        "    Copyable where conforms_to(T, Copyable)"
        ' else "wrapped type must be copyable",\n'
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Ordered[T: Movable](\n"
        "    Copyable where conforms_to(\n"
        "        T, Copyable\n"
        '    ) else "wrapped type must be copyable",\n'
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ========================================== #
# Alias/comptime and function types
# ========================================== #


def test_alias_where_clause_else_message():
    source = (
        "comptime P[T: AnyType] where conforms_to(T, Copyable)"
        ' else "T must be copyable" = T'
    )
    expected = (
        "comptime P[T: AnyType] where conforms_to(\n"
        "    T, Copyable\n"
        ') else "T must be copyable" = T\n'
    )
    assert_mojo_format(source, expected)


def test_fn_type_where_clause_else_message():
    source = (
        "comptime K = def[w: Int]() thin -> None"
        ' where w > 0 else "width must be positive"'
    )
    expected = (
        "comptime K = def[w: Int]() thin -> None where (\n"
        "    w > 0\n"
        ') else "width must be positive"\n'
    )
    assert_mojo_format(source, expected)
