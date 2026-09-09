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

"""Formatter tests for trailing `where` clauses on function types.

A `thin` function type carries its constraints after the result type, in the
same position a function declaration carries them. The clause binds to the
innermost function type, so a declaration-level `where` following a
function-type result needs that result parenthesized -- the formatter has to
preserve that distinction rather than normalize it away.
"""

from tests.util import assert_mojo_format


# ============================================================ #
# Basic forms
# ============================================================ #


def test_fn_type_trailing_where_in_alias():
    source = "comptime Kernel = def[w: Int](Int) thin -> None where w > 0"
    expected = "comptime Kernel = def[w: Int](Int) thin -> None where w > 0\n"
    assert_mojo_format(source, expected)


def test_fn_type_trailing_where_multiple_clauses():
    source = (
        "comptime Bounded = def[w: Int]() thin -> None where w > 0 where w < 64"
    )
    expected = (
        "comptime Bounded = def[w: Int]() thin -> None where w > 0 where w < 64\n"
    )
    assert_mojo_format(source, expected)


def test_fn_type_trailing_where_with_message():
    source = 'comptime K = def[w: Int]() thin -> None where (w > 0, "positive")'
    expected = (
        'comptime K = def[w: Int]() thin -> None where (w > 0, "positive")\n'
    )
    assert_mojo_format(source, expected)


def test_fn_type_trailing_where_as_parameter_type():
    source = (
        "def apply[F: def[w: Int](Int) thin -> None where w > 0](x: Int):\n"
        "    F[4](x)\n"
    )
    assert_mojo_format(source, source)


# ============================================================ #
# Binding: innermost function type wins
# ============================================================ #


def test_fn_type_where_binds_to_result_type():
    """Unparenthesized: the clause belongs to the returned function type."""
    source = (
        "def make[n: Int]() -> def() thin -> None where n > 0:\n"
        "    return lambda: None\n"
    )
    assert_mojo_format(source, source)


def test_fn_type_where_parenthesized_binds_to_decl():
    """Parenthesized: the clause belongs to the declaration.

    The parentheses are load-bearing here, so they must survive formatting.
    """
    source = (
        "def make[n: Int]() -> (def() thin -> None) where n > 0:\n"
        "    return lambda: None\n"
    )
    assert_mojo_format(source, source)

