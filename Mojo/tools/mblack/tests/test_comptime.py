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

import pytest
import mblack.parsing
from tests.util import assert_mojo_format, mojo_format_str


def test_global_comptime():
    source = (
        "comptime   b =  6\n"
    )
    expected = (
        "comptime b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_nested_comptime():
    source = (
        "struct Foo:\n"
        "    comptime  b =  6\n"
    )
    expected = (
        "struct Foo:\n"
        "    comptime b = 6\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if():
    source = (
        "def foo[a: Bool]():\n"
        "    comptime if a:\n"
        "            var x: Int\n"
    )
    expected = (
        "def foo[a: Bool]():\n"
        "    comptime if a:\n"
        "        var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if_elif_else():
    source = (
        "def foo[a: Bool,  b: Bool]():\n"
        "    comptime if a:\n"
        "            var x: Int\n"
        "    elif b:\n"
        "                var y: Int\n"
        "    else:\n"
        "        var z: Int\n"
    )
    expected = (
        "def foo[a: Bool, b: Bool]():\n"
        "    comptime if a:\n"
        "        var x: Int\n"
        "    elif b:\n"
        "        var y: Int\n"
        "    else:\n"
        "        var z: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_for():
    source = (
        "def foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "            print(i)\n"
    )
    expected = (
        "def foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        print(i)\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_for_nested_if():
    source = (
        "def foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        comptime if True:\n"
        "                    print(i)\n"
    )
    expected = (
        "def foo[a: Int]():\n"
        "    comptime for i in range(a):\n"
        "        comptime if True:\n"
        "            print(i)\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_if_extra_whitespace():
    """Extra whitespace between comptime and if should be normalized."""
    source = (
        "def foo[x: Int]():\n"
        "    comptime        if x > 0:\n"
        "        var y: Int\n"
    )
    expected = (
        "def foo[x: Int]():\n"
        "    comptime if x > 0:\n"
        "        var y: Int\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_on_separate_line():
    """comptime on separate line from if is invalid syntax."""
    source = (
        "def foo[x: Int]():\n"
        "    comptime\n"
        "    if x > 0:\n"
        "        var y: Int\n"
    )
    with pytest.raises(mblack.parsing.InvalidInput):
        mojo_format_str(source)


def test_comptime_expr_rhs_of_binop():
    """comptime(expr) on the right-hand side of a binary operator."""
    source = "def foo():\n" "    var y = 1 + comptime(1 * 2)\n"
    expected = "def foo():\n" "    var y = 1 + comptime (1 * 2)\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_lhs_of_binop():
    """comptime(expr) on the left-hand side of a binary operator."""
    source = "def foo():\n" "    var y = comptime(1 * 2) + 1\n"
    expected = "def foo():\n" "    var y = comptime (1 * 2) + 1\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_as_argument():
    """comptime(expr) as a function argument."""
    source = "def foo():\n" "    print(comptime(42))\n"
    expected = "def foo():\n" "    print(comptime (42))\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_multiple():
    """Multiple comptime(expr) in one expression."""
    source = "def foo():\n" "    var z = comptime(1) + comptime(2)\n"
    expected = "def foo():\n" "    var z = comptime (1) + comptime (2)\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_in_list_literal():
    """comptime(expr) inside a list literal."""
    source = "def foo():\n" "    var x = [comptime(1), comptime(2)]\n"
    expected = "def foo():\n" "    var x = [comptime (1), comptime (2)]\n"
    assert_mojo_format(source, expected)


def test_comptime_expr_in_if_condition():
    """comptime(expr) used as a condition in an if statement."""
    source = (
        "def foo():\n"
        "    if comptime(1 + 2 == 3):\n"
        "        pass\n"
    )
    expected = (
        "def foo():\n"
        "    if comptime (1 + 2 == 3):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_comptime_illegal_keyword():
    """comptime with illegal keyword should be rejected."""
    source = (
        "def foo():\n"
        "    comptime try:\n"
        "        var x: Int\n"
    )
    with pytest.raises(mblack.parsing.InvalidInput):
        mojo_format_str(source)


def test_multiple_comptime_same_scope():
    """Multiple comptime declarations at the same scope level."""
    source = (
        "comptime X = Int\n"
        "comptime Y = Float64\n"
    )
    assert_mojo_format(source, source)


def test_comptime_assert():
    """comptime assert statement is formatted correctly."""
    source = (
        "def foo():\n"
        '    comptime   assert   True,   "msg"\n'
    )
    expected = (
        "def foo():\n"
        '    comptime assert True, "msg"\n'
    )
    assert_mojo_format(source, expected)
    