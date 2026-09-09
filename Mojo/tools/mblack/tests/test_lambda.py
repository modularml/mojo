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

# Tests for Mojo `lambda` expressions. A Mojo lambda mirrors a `def` signature
# (parenthesized typed args, plus optional metaparams / effects / captures /
# return type) with an expression body. A lambda that fits stays inline; an
# overlong one wraps via the standard expression-splitting path.

import mblack.parsing
import pytest
from tests.util import assert_mojo_format, mojo_format_str


def test_no_captures_explicit_return():
    """A non-capturing lambda with an explicit return type is left unchanged."""
    source = "def main():\n    var f = lambda (x: Int) -> Int: x + 1\n"
    assert_mojo_format(source, source)


def test_elided_return_none_body():
    """An elided return type (None default) with a None body is unchanged."""
    source = "def main():\n    var f = lambda (x: Int) {}: None\n"
    assert_mojo_format(source, source)


def test_bare_lambda():
    """The fully bare `lambda: expr` form (Python overlap) is unchanged."""
    source = "def main():\n    var f = lambda: None\n"
    assert_mojo_format(source, source)


def test_empty_arg_list():
    """An empty parenthesized arg list is unchanged."""
    source = "def main():\n    var f = lambda () -> Int: 1\n"
    assert_mojo_format(source, source)


def test_no_arg_list_with_captures():
    """The arg list may be omitted entirely, leaving just a capture list."""
    source = "def main():\n    var f = lambda {} -> Int: 1\n"
    assert_mojo_format(source, source)


def test_no_arg_list_no_captures():
    """The arg list and capture list may both be omitted."""
    source = "def main():\n    var f = lambda -> Int: 1\n"
    assert_mojo_format(source, source)


def test_reformats_no_arg_list_spacing():
    """A missing space before an omitted-arg-list capture list is inserted."""
    source = "def main():\n    var f = lambda{}->Int:1\n"
    expected = "def main():\n    var f = lambda {} -> Int: 1\n"
    assert_mojo_format(source, expected)


def test_effects():
    """`raises` between the argument list and capture list is unchanged."""
    source = "def main():\n    var f = lambda (x: Int) raises {} -> Int: x + 1\n"
    assert_mojo_format(source, source)


@pytest.mark.parametrize("conv", ["imm", "mut", "var", "ref"])
def test_arg_conventions(conv):
    """Each argument convention formats with a single space after it."""
    source = f"def main():\n    var f = lambda ({conv} x: Int) {{}} -> Int: x + 1\n"
    assert_mojo_format(source, source)


def test_capture_by_mut():
    """A `{mut z}` capture of an outer variable is unchanged."""
    source = (
        "def main():\n"
        "    var z = 3\n"
        "    var f = lambda (x: Int) {mut z} -> Int: x + z\n"
    )
    assert_mojo_format(source, source)


def test_omitted_capture_list_reads_free_var():
    """An omitted capture list (imm-captures the free `z`) is unchanged."""
    source = (
        "def main():\n"
        "    var z = 3\n"
        "    var f = lambda (x: Int) -> Int: x + z\n"
    )
    assert_mojo_format(source, source)


def test_everything():
    """Metaparams + owned arg + effects + capture-all + return type."""
    source = (
        "def main() raises:\n"
        "    var z = 3\n"
        "    var f = lambda [N: Int](var x: Int) raises {mut} -> Int: x + N + z\n"
    )
    assert_mojo_format(source, source)


def test_adds_space_after_lambda_keyword():
    """A missing space after `lambda` before `(` is inserted."""
    source = "def main():\n    var f = lambda(x: Int) -> Int: x + 1\n"
    expected = "def main():\n    var f = lambda (x: Int) -> Int: x + 1\n"
    assert_mojo_format(source, expected)


def test_adds_space_after_lambda_keyword_metaparams():
    """A missing space after `lambda` before `[` is inserted."""
    source = "def main():\n    var f = lambda[N: Int](x: Int) {} -> Int: x + N\n"
    expected = "def main():\n    var f = lambda [N: Int](x: Int) {} -> Int: x + N\n"
    assert_mojo_format(source, expected)


def test_reformats_tight_spacing():
    """Tight spacing is normalized: args, capture list, arrow, and body."""
    source = "def main():\n    var f = lambda (x:Int){}->Int:x+1\n"
    expected = "def main():\n    var f = lambda (x: Int) {} -> Int: x + 1\n"
    assert_mojo_format(source, expected)


def test_long_lambda_wraps_via_standard_expression_splitting():
    """A lambda that exceeds the line width is not kept on one line: it wraps
    through the normal expression-splitting path (no lambda-specific splitting).
    The RHS is parenthesized, the arg list wraps, and the body splits on its
    operators."""
    source = (
        "def main():\n"
        "    var f = lambda (some_long_arg_name: Int, another_long_one: Int)"
        " -> Int: some_long_arg_name + another_long_one + 123456789\n"
    )
    expected = (
        "def main():\n"
        "    var f = (\n"
        "        lambda (\n"
        "            some_long_arg_name: Int, another_long_one: Int\n"
        "        ) -> Int: some_long_arg_name\n"
        "        + another_long_one\n"
        "        + 123456789\n"
        "    )\n"
    )
    assert_mojo_format(source, expected)


def test_magic_trailing_comma_explodes_arg_list():
    """Mojo lambda args ride the `def`-parameter grammar, so the magic trailing
    comma explodes the arg list one-per-line (unlike Python lambda args in
    `varargslist`, where black ignores the magic comma)."""
    source = "def main():\n    var f = lambda (x: Int,) {} -> Int: x\n"
    expected = (
        "def main():\n"
        "    var f = lambda (\n"
        "        x: Int,\n"
        "    ) {} -> Int: x\n"
    )
    assert_mojo_format(source, expected)


def test_lambda_in_comprehension_if():
    """A Mojo lambda in a comprehension's `if` (the `old_lambdef` position) is
    unchanged. Semantically dead (a closure is not Bool-convertible) but
    grammatically an expression position, so it must format."""
    source = (
        "def main():\n"
        "    var l = [x for x in y if lambda (x: Int) {} -> Int: x]\n"
    )
    assert_mojo_format(source, source)


def test_lambda_in_comprehension_if_reformats():
    """Tight spacing normalizes in the comprehension-`if` position exactly as
    it does in the plain-expression position (spacing rule fires under
    `old_lambdef` too)."""
    source = "def main():\n    var l = [x for x in y if lambda(x:Int){}->Int:x]\n"
    expected = (
        "def main():\n"
        "    var l = [x for x in y if lambda (x: Int) {} -> Int: x]\n"
    )
    assert_mojo_format(source, expected)


def test_lambda_in_generator_if():
    """A Mojo lambda in a generator expression's `if` (the `comp_if` position,
    reached from a call's argument list) is unchanged."""
    source = (
        "def main():\n"
        "    f(x for x in y if lambda [N: Int](x: Int) raises {mut} -> Int: x)\n"
    )
    assert_mojo_format(source, source)


def test_lambda_as_comprehension_iterable():
    """A Mojo lambda as a parenthesized generator's iterable (the
    `testlist_safe` position) is unchanged."""
    source = "def main():\n    var g = (x for x in lambda (y: Int) -> Int: y)\n"
    assert_mojo_format(source, source)


def test_lambda_in_comprehension_if_keeps_trailing_if():
    """The lambda's body stays `old_test`, which cannot be a bare conditional,
    so a trailing `if` stays the comprehension's filter instead of becoming a
    ternary that would swallow it."""
    source = (
        "def main():\n"
        "    var l = [x for x in y if lambda {} -> Int: x if z]\n"
    )
    assert_mojo_format(source, source)


def test_lambda_in_comprehension_if_rejects_bare_conditional_body():
    """A bare (unparenthesized) conditional as the lambda body in a
    comprehension `if` is a parse error — the `old_test` body cannot be a
    ternary, exactly as for Python lambdas in the same position."""
    source = (
        "def main():\n"
        "    var l = [x for x in y if lambda {} -> Int: x if a else b]\n"
    )
    with pytest.raises(mblack.parsing.InvalidInput):
        mojo_format_str(source)


def test_lambda_in_comprehension_if_with_trailing_for():
    """A lambda in a comprehension `if` followed by another `for` clause still
    parses (neither lambda body production can contain a `for`)."""
    source = (
        "def main():\n"
        "    var l = [x for x in y if lambda {} -> Int: x for z in w]\n"
    )
    assert_mojo_format(source, source)


def test_lambda_nested_in_old_lambdef_body():
    """An `old_lambdef` body is itself `old_test`, so a Mojo lambda nests as
    the body of a lambda in a comprehension `if`."""
    source = (
        "def main():\n"
        "    var l = [x for x in y if lambda: lambda (z: Int) -> Int: z]\n"
    )
    assert_mojo_format(source, source)


def test_variadic_args():
    """`*args` / `**kwargs` ride the `def`-parameter grammar inside the
    parenthesized lambda arg list."""
    source = (
        "def main():\n"
        "    var f = lambda (*args: Int) {} -> Int: 0\n"
        "    var g = lambda (var **kwargs: Int) {} -> Int: 0\n"
        "    var h = lambda (x: Int, *args: Int, var **kwargs: Int) {} -> Int: x\n"
    )
    assert_mojo_format(source, source)
