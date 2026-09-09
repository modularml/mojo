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

"""Tests for the optional message on `where` clauses.

Mojo `where` clauses accept an optional message written
`where (condition, "message")`. The message lives inside the parentheses, so
no grammar changes are needed beyond the existing `where <expression>` rule --
the parenthesized `(condition, "message")` is an ordinary tuple expression.
"""

from tests.util import assert_mojo_format

# ======================== #
# Fns with where messages
# ======================== #


def test_fn_where_clause_message():
    source = (
        'def gated[sc: Int]() where (sc > 1, "scaling factor must be'
        ' greater than 1"): pass'
    )
    expected = (
        'def gated[sc: Int]() where (sc > 1, "scaling factor must be'
        ' greater than 1"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_short_where_clause_message():
    source = 'def gated[x: Bool]() where (x, "x must hold"): pass'
    expected = 'def gated[x: Bool]() where (x, "x must hold"):\n    pass\n'
    assert_mojo_format(source, expected)


def test_multiple_fn_where_clauses_with_messages():
    source = (
        "def multi[a: Int, b: Int]() "
        'where (a > 0, "a positive") where (b > 0, "b positive"): pass'
    )
    expected = (
        "def multi[\n"
        "    a: Int, b: Int\n"
        ']() where (a > 0, "a positive") where (b > 0, "b positive"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_message_adjacent_literals():
    """Adjacent string literals inside the message are merged by the string
    transformer when they fit on one line."""
    source = 'def concat[x: Int]() where (x > 0, "part one " "part two"): pass'
    expected = (
        'def concat[x: Int]() where (x > 0, "part one part two"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_message_wraps_when_too_long():
    """A message too long for one line wraps as adjacent string literals
    (the only wrapping the parser accepts here). MOCO-4446."""
    source = (
        "def gated[n: Int]() where (n > 0,"
        ' "the scaling factor must be greater than one for this operation'
        ' to be well defined"):\n'
        "    pass\n"
    )
    expected = (
        "def gated[\n"
        "    n: Int\n"
        "]() where (\n"
        "    n > 0,\n"
        '    "the scaling factor must be greater than one for this operation'
        ' to be well"\n'
        '    " defined",\n'
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_message_prewrapped_adjacent_literals_too_long():
    """Author-supplied adjacent-literal split points are preserved when the
    merged form is still too long. MOCO-4446."""
    source = (
        "def gated[n: Int]() where (\n"
        "    n > 0,\n"
        '    "the scaling factor must be greater"\n'
        '    " than one for this operation to be well defined",\n'
        "):\n"
        "    pass\n"
    )
    expected = (
        "def gated[\n"
        "    n: Int\n"
        "]() where (\n"
        "    n > 0,\n"
        '    "the scaling factor must be greater"\n'
        '    " than one for this operation to be well defined",\n'
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_fn_where_clause_long_message_without_spaces():
    """A long message with no splittable space stays intact while the clause
    explodes across lines. MOCO-4446."""
    source = (
        "def gated[n: Int]() where (n > 0,"
        ' "the_scaling_factor_must_be_greater_than_one_for_this_operation"'
        "):\n"
        "    pass\n"
    )
    expected = (
        "def gated[\n"
        "    n: Int\n"
        "]() where (\n"
        "    n > 0,\n"
        '    "the_scaling_factor_must_be_greater_than_one_for_this_operation",\n'
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================ #
# Structs with where messages
# ============================ #


def test_struct_where_clause_message():
    source = 'struct Gated[N: Int] where (N > 0, "N must be positive"): pass'
    expected = (
        'struct Gated[N: Int] where (N > 0, "N must be positive"):\n'
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ===================================== #
# Conformance lists with where messages
# ===================================== #


def test_conditional_conformance_message():
    source = (
        "struct Wrapped[T: Movable](\n"
        "    Copyable where (conforms_to(T, Copyable),"
        ' "wrapped type must be copyable"),\n'
        "):\n"
        "    pass\n"
    )
    expected = source
    assert_mojo_format(source, expected)


def test_conditional_conformance_message_sorts_with_trait():
    """The conformance-list sorter reorders traits alphabetically. The
    message lives inside the trait's `where (...)` parentheses, so it moves
    with its trait and stays correctly attached."""
    source = (
        "struct Ordered[T: Movable](\n"
        "    Movable,\n"
        "    Copyable where (conforms_to(T, Copyable),"
        ' "wrapped type must be copyable"),\n'
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Ordered[T: Movable](\n"
        "    Copyable where (conforms_to(T, Copyable),"
        ' "wrapped type must be copyable"),\n'
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ===================================== #
# Alias/comptime with where messages
# ===================================== #


def test_alias_where_clause_message():
    source = (
        "comptime P[T: AnyType] where (conforms_to(T, Copyable),"
        ' "T must be copyable") = T'
    )
    expected = (
        "comptime P[T: AnyType] where (\n"
        "    conforms_to(T, Copyable),\n"
        '    "T must be copyable",\n'
        ") = T\n"
    )
    assert_mojo_format(source, expected)
