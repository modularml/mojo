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

"""Formatter tests for trailing `where` clauses on `comptime` declarations.

The trailing `where` clause follows the alias type (if any) and precedes the
`=` initializer (if any), mirroring the position used by functions and
structs. Multiple `where` clauses chain as separate keywords.
"""

from tests.util import assert_mojo_format, mojo_format_str


# ============================================================ #
# `comptime` form
# ============================================================ #


def test_comptime_alias_trailing_where_no_type():
    source = "comptime Positive[n: Int] where n > 0 = n"
    expected = "comptime Positive[n: Int] where n > 0 = n\n"
    assert_mojo_format(source, expected)


def test_comptime_alias_trailing_where_with_type():
    source = "comptime Positive[n: Int] : Int where n > 0 = n"
    expected = "comptime Positive[n: Int]: Int where n > 0 = n\n"
    assert_mojo_format(source, expected)


def test_comptime_trait_associated_parametric_alias_no_initializer():
    """Trait-associated parametric `comptime` member with a trailing where.

    No `=` initializer; only the type and constraint follow the parameters.
    Wrapped in a trait so the body is a valid declaration site.
    """
    source = (
        "trait HasEven:\n"
        "    comptime Assoc[T: AnyType] : AnyType where conforms_to(T, AnyType)\n"
    )
    expected = (
        "trait HasEven:\n"
        "    comptime Assoc[T: AnyType]: AnyType where conforms_to(T, AnyType)\n"
    )
    assert_mojo_format(source, expected)


# ============================================================ #
# Line breaking
# ============================================================ #


def test_alias_trailing_where_long_predicate_wraps():
    """A long trailing `where` clause wraps onto its own line."""
    source = (
        "comptime Constrained[n: Int, m: Int]: Int where "
        "n > 0 and m > 0 and n + m < 1024 and n * n > 0 and m * m > 0 = n + m"
    )
    expected = (
        "comptime Constrained[n: Int, m: Int]: Int where (\n"
        "    n > 0 and m > 0 and n + m < 1024 and n * n > 0 and m * m > 0\n"
        ") = n + m\n"
    )
    assert_mojo_format(source, expected)


def test_alias_trailing_where_long_param_list_wraps_too():
    """Long parameter list and a trailing `where` clause both wrap.

    Mojo's formatter wraps the parameter list onto multiple lines and
    parenthesizes the `where` predicate when the resulting comptime header is
    too long to fit on a single line.
    """
    source = (
        "comptime VeryLongAliasWithManyParams["
        "first_param: Int, second_param: Int, third_param: Int, "
        "fourth_param: Int] where first_param > 0 and second_param > 0 = "
        "first_param + second_param + third_param + fourth_param"
    )
    expected = (
        "comptime VeryLongAliasWithManyParams[\n"
        "    first_param: Int, second_param: Int, third_param: Int,"
        " fourth_param: Int\n"
        "] where (\n"
        "    first_param > 0 and second_param > 0\n"
        ") = first_param + second_param + third_param + fourth_param\n"
    )
    assert_mojo_format(source, expected)


# ============================================================ #
# Idempotency & comments
# ============================================================ #


def test_alias_trailing_where_idempotent():
    """Formatting an already-formatted comptime is a no-op."""
    source = "comptime Positive[n: Int]: Int where n > 0 = n"
    formatted_once = mojo_format_str(source)
    assert_mojo_format(formatted_once, formatted_once)


def test_alias_trailing_where_with_trailing_comment():
    source = (
        "comptime Positive[n: Int]: Int where n > 0 = n  # must be positive"
    )
    expected = (
        "comptime Positive[n: Int]: Int where n > 0 = n  # must be positive\n"
    )
    assert_mojo_format(source, expected)
