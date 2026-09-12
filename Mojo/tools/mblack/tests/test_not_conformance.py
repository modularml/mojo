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

"""Tests for formatting the `not Trait` conformance-list entry.

`not Trait` is the opt-out spelling of `Trait where False`, which
`test_conditional_conformance` covers. Unlike the `where ... else` message it
needs no grammar rule of its own: a conformance list is an `arglist`, whose
`argument` is a `test`, and `not_test: 'not' not_test | comparison` already
admits the prefix `not`. These tests pin that, and that a `not` entry wraps and
splits like any other entry.
"""

from tests.util import assert_mojo_format

# ============================ #
# Single `not` entry
# ============================ #


def test_single_not_conformance():
    source = "struct Opaque(not Movable): pass"
    expected = "struct Opaque(not Movable):\n    pass\n"
    assert_mojo_format(source, expected)


def test_not_conformance_is_not_reparenthesized():
    """`not` is a prefix operator, so no parentheses are introduced."""
    source = "struct Opaque(not Deinitable):\n    pass\n"
    assert_mojo_format(source, source)


def test_not_trait_composition():
    source = "struct Opaque(not (Readable & Writable)):\n    pass\n"
    assert_mojo_format(source, source)


# ============================ #
# Mixed with other entry forms
# ============================ #


def test_not_mixed_with_plain_and_conditional_entries():
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    not Movable,\n"
        "    Writable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, source)


def test_short_mixed_list_stays_on_one_line():
    source = "struct Pair(not Movable, Writable): pass"
    expected = "struct Pair(not Movable, Writable):\n    pass\n"
    assert_mojo_format(source, expected)


def test_long_list_with_not_entry_splits():
    """A long list wraps, and the `not` entry sorts under the trait it names."""
    source = (
        "struct VeryLongStructNameHere[T: Movable](not Movable, Writable,"
        " Stringable, Representable, Boolable): pass"
    )
    expected = (
        "struct VeryLongStructNameHere[T: Movable](\n"
        "    Boolable, not Movable, Representable, Stringable, Writable\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_not_entry_sorts_under_its_trait_name():
    source = "struct Pair(Writable, not Movable, Boolable): pass"
    expected = "struct Pair(Boolable, not Movable, Writable):\n    pass\n"
    assert_mojo_format(source, expected)


def test_not_entry_with_magic_trailing_comma_explodes():
    source = "struct Pair(not Movable, Writable,): pass"
    expected = (
        "struct Pair(\n"
        "    not Movable,\n"
        "    Writable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
