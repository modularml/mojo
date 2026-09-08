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

"""Tests for the // (positional-only separator) sigil formatting in Mojo."""

from tests.util import assert_mojo_format


def test_doubleslash_on_own_line_simple():
    """Test that // is placed on its own line in multi-line function signatures."""
    source = "def foo[a: Int, //, b: Int, *, c: Int,](): pass"
    expected = """\
def foo[
    a: Int,
    //,
    b: Int,
    *,
    c: Int,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_on_own_line_with_types():
    """Test // with typed parameters is placed on its own line."""
    source = "def bar[x: Int, y: String, //, z: Bool,](): pass"
    expected = """\
def bar[
    x: Int,
    y: String,
    //,
    z: Bool,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_with_defaults():
    """Test // with default values is placed on its own line."""
    source = "def baz[a: Int = 1, b: Int = 2, //, c: Int = 3,](): pass"
    expected = """\
def baz[
    a: Int = 1,
    b: Int = 2,
    //,
    c: Int = 3,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_inline_fits():
    """Test that // stays inline when the signature fits on one line."""
    source = "def f[a: Int, //](): pass"
    expected = """\
def f[a: Int, //]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_and_star_together():
    """Test that both // and * are placed on their own lines."""
    source = "def combined[pos1: Int, pos2: Int, //, normal: Int, *, kwonly: Int,](): pass"
    expected = """\
def combined[
    pos1: Int,
    pos2: Int,
    //,
    normal: Int,
    *,
    kwonly: Int,
]():
    pass
"""
    assert_mojo_format(source, expected)


def test_doubleslash_in_long_signature():
    """Test // in a long signature gets its own line."""
    source = "def long_function_name[very_long_parameter_name: Int, another_long_param: String, //, yet_another_param: Bool,](): pass"
    expected = """\
def long_function_name[
    very_long_parameter_name: Int,
    another_long_param: String,
    //,
    yet_another_param: Bool,
]():
    pass
"""
    assert_mojo_format(source, expected)
