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

"""Tests for `mblack -t mojo` without --preview.

The main test suite uses MOJO_MODE (preview=True) to match `mojo format`.
These tests verify the non-preview path — reachable via `mblack -t mojo`
without --preview — for the features where preview changes behavior.
"""

from tests.util import MOJO_MODE_NO_PREVIEW, assert_mojo_format


def test_implicit_string_concat_kept_separate():
    """preview=True merges adjacent strings; preview=False keeps them separate."""
    source = (
        "def main():\n"
        '    var long_text = "first part" \\\n'
        '                     " second part"\n'
    )
    expected = (
        "def main():\n"
        '    var long_text = "first part" " second part"\n'
    )
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)


def test_implicit_string_concat_chained_kept_separate():
    """Chained adjacent strings also stay separate without preview."""
    source = (
        "def main():\n"
        '    var s = "one"\n'
        '             " two"\n'
        '             " three"\n'
    )
    expected = (
        "def main():\n"
        '    var s = "one" " two" " three"\n'
    )
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)


def test_block_trailing_newline_preserved():
    """Non-preview keeps blank lines between statements inside a block."""
    source = (
        "def foo():\n"
        "    x = 1\n"
        "\n"
        "    y = 2\n"
    )
    assert_mojo_format(source, source, mode=MOJO_MODE_NO_PREVIEW)


def test_percent_comment_no_space():
    """preview=False allows #% without a space; preview=True inserts one."""
    source = "def main():\n    x = 1  #% some comment\n"
    assert_mojo_format(source, source, mode=MOJO_MODE_NO_PREVIEW)


def test_comptime_if():
    """comptime if formats correctly without preview."""
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
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)


def test_keyword_spacing():
    """Argument convention keywords format correctly without preview."""
    source = (
        "struct Foo:\n"
        "    def __init__(out self, mut v:Int, imm x:Int): pass\n"
    )
    expected = (
        "struct Foo:\n"
        "    def __init__(out self, mut v: Int, imm x: Int):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)


def test_parametric_alias():
    """Parametric aliases format correctly without preview."""
    source = "comptime addOne[x: Int] : Int = x + 1"
    expected = "comptime addOne[x: Int]: Int = x + 1\n"
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)


def test_var_type_annotation():
    """Type annotations format correctly without preview."""
    source = "def foo():\n    var x:Int = 1\n"
    expected = "def foo():\n    var x: Int = 1\n"
    assert_mojo_format(source, expected, mode=MOJO_MODE_NO_PREVIEW)
