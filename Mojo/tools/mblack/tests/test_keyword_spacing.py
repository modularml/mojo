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

from mblack.parsing import InvalidInput
from tests.util import assert_mojo_format, mojo_format_str


def test_unknown_convention_raises_error():
    source = "def foo(aaa self): pass"
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mojo_format_str(source)


def test_inout_raises_error():
    """inout has been removed from Mojo; the formatter should reject it."""
    source = "def method(inout self): pass"
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mojo_format_str(source)


def test_borrowed_raises_error():
    """borrowed has been removed from Mojo; the formatter should reject it."""
    source = "def method(borrowed self): pass"
    with pytest.raises(InvalidInput, match="unknown argument convention"):
        mojo_format_str(source)


def test_var_keyword_spacing_simple():
    source = "struct Foo:\n    def method(var self): pass"
    expected = (
        "struct Foo:\n"
        "    def method(var self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_with_type():
    source = "def func(var x: Int): pass"
    expected = (
        "def func(var x: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_var_keyword_spacing_mixed():
    source = "struct Foo:\n    def method(var self, y: Int): pass"
    expected = (
        "struct Foo:\n"
        "    def method(var self, y: Int):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing():
    source = "struct Foo:\n    def __init__(out self, mut v:Int, imm x:Int): pass"
    expected = (
        "struct Foo:\n"
        "    def __init__(out self, mut v: Int, imm x: Int):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_imm_keyword_spacing():
    """`imm` is a contextual keyword and must be spaced like other argument conventions."""
    source = "struct Foo:\n    def __init__(out self, mut v:Int, imm x:Int): pass"
    expected = (
        "struct Foo:\n"
        "    def __init__(out self, mut v: Int, imm x: Int):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_contextual_keyword_spacing_variadics():
    source = (
        "struct Foo:\n"
        "    def __init__(out self, mut *v: Int, var **kwargs: Int): pass"
    )
    expected = (
        "struct Foo:\n"
        "    def __init__(out self, mut *v: Int, var **kwargs: Int):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)
