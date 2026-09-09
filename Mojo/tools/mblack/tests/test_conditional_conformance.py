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

"""Tests for formatting conditional trait conformances in struct definitions.

Conditional conformance allows a struct to conform to a trait only when certain
conditions are met. The syntax is:
    TraitName where condition

For example:
    struct Wrapper[T: Movable](
        Copyable where conforms_to(T, Copyable),
        Movable,
    ):
"""

import pytest

from tests.util import assert_mojo_format

# ============================================ #
# Basic conditional conformance in struct defs
# ============================================ #


def test_simple_conditional_conformance():
    """Test a simple conditional conformance with conforms_to."""
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_conditional_conformances():
    """Test struct with multiple conditional conformances."""
    source = (
        "trait Countable:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Multi[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Countable where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait Countable:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Multi[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Countable where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_with_and():
    """Test conditional conformance with 'and' operator."""
    source = (
        "trait TraitA:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Wrapper[T: Movable](\n"
        "    TraitA where conforms_to(T, Copyable) and conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait TraitA:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Wrapper[T: Movable](\n"
        "    Movable,\n"
        "    TraitA where conforms_to(T, Copyable) and conforms_to(T, Intable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_with_or():
    """Test conditional conformance with 'or' operator."""
    source = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Wrapper[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Wrapper[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_conditional_conformance_single_line():
    """Test compact conditional conformance that fits on one line."""
    source = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](A where conforms_to(T, B), M): pass"
    )
    expected = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](A where conforms_to(T, B), M):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================================ #
# Long conditional conformances that need wrapping
# ================================================ #


def test_long_conditional_conformance():
    """Test conditional conformance that needs line wrapping."""
    source = (
        "struct VeryLongWrapper[T: Deinitable & Movable]("
        "Copyable where conforms_to(T, Copyable) and conforms_to(T, Intable) and conforms_to(T, Writable), "
        "Movable): pass"
    )
    expected = (
        "struct VeryLongWrapper[T: Deinitable & Movable](\n"
        "    Copyable where (\n"
        "        conforms_to(T, Copyable)\n"
        "        and conforms_to(T, Intable)\n"
        "        and conforms_to(T, Writable)\n"
        "    ),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_long_conditional_conformances():
    """Test multiple conditional conformances that need line wrapping."""
    source = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedB:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Diamond[T: Movable](\n"
        "Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "DerivedA where conforms_to(T, Copyable),\n"
        "DerivedB where conforms_to(T, Intable),\n"
        "Movable,\n"
        "): pass"
    )
    expected = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedB:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Diamond[T: Movable](\n"
        "    Base where conforms_to(T, Copyable) or conforms_to(T, Intable),\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ================================================ #
# Mixed conditional and unconditional conformances
# ================================================ #


def test_mixed_conditional_unconditional():
    """Test struct with both conditional and unconditional conformances."""
    source = (
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedB:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Mixed[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedB:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Mixed[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    DerivedB,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_explicit_ancestor_with_conditional():
    """Test explicit ancestor conformance with conditional."""
    source = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Explicit[T: Movable](\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    Base,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    # Formatter sorts all conformances alphabetically, including conditional ones
    expected = (
        "trait Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait DerivedA:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Explicit[T: Movable](\n"
        "    Base,\n"
        "    DerivedA where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================== #
# Conditional conformance with T
# ============================== #


def test_conditional_conformance_with_type():
    """Test conditional conformance referencing the generic parameter.
    """
    source = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct Wrapper[T: Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ======================================== #
# Complex struct definitions with generics
# ======================================== #


def test_complex_generic_with_conditional():
    """Test complex generic struct with conditional conformance."""
    source = (
        "struct Container[T: Deinitable & Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Deinitable,\n"
        "    Movable,\n"
        "):\n"
        "    var value: Self.T\n"
    )
    expected = (
        "struct Container[T: Deinitable & Movable](\n"
        "    Copyable where conforms_to(T, Copyable),\n"
        "    Deinitable,\n"
        "    Movable,\n"
        "):\n"
        "    var value: Self.T\n"
    )
    assert_mojo_format(source, expected)


# ====================================== #
# Whitespace normalization in conditions
# ====================================== #


def test_whitespace_normalization_in_condition():
    """Test that extra whitespace in conditions is normalized."""
    source = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](\n"
        "    A where   conforms_to(T,   B),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](\n"
        "    A where conforms_to(T, B),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_condition_with_complex_expression():
    """Test conditional conformance with complex boolean expression."""
    source = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait C:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait D:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](\n"
        "    A where (conforms_to(T, B) and conforms_to(T, C)) or conforms_to(T, D),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    # Expression fits on one line, so no wrapping is applied
    expected = (
        "trait A:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait B:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait C:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait D:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait M:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct W[T: M](\n"
        "    A where (conforms_to(T, B) and conforms_to(T, C)) or conforms_to(T, D),\n"
        "    M,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================================================== #
# Sorting correctness: where clauses must stay with their traits
# ============================================================== #


def test_sorting_keeps_where_clause_with_trait():
    """Sorting must move the where clause together with its trait name.

    Previously, sorting only operated on bare NAME leaves, so conditional
    conformances were either skipped or could get detached from their
    where clause.
    """
    source = (
        "trait Alpha:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait Zebra:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Foo[T: Movable](\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "    Alpha,\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    # Alpha < Movable < Zebra — and the where clause stays on Zebra.
    expected = (
        "trait Alpha:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait Zebra:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Foo[T: Movable](\n"
        "    Alpha,\n"
        "    Movable,\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_sorting_multiple_conditional_conformances():
    """Multiple conditional conformances are sorted with their where clauses."""
    source = (
        "trait Alpha:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait Zebra:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Foo[T: Movable](\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "    Alpha where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "trait Alpha:\n"
        "    pass\n"
        "\n"
        "\n"
        "trait Zebra:\n"
        "    pass\n"
        "\n"
        "\n"
        "struct Foo[T: Movable](\n"
        "    Alpha where conforms_to(T, Intable),\n"
        "    Movable,\n"
        "    Zebra where conforms_to(T, Copyable),\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
