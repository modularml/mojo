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

from tests.util import assert_mojo_format

# List of keywords for Python and Mojo. Mojo keywords are allowed as member
# function names, and as the current mojo format implementation is based on
# mblack check that Python keywords are as well.
# NOTE: Keep this list in sync with lists of keywords in the source.
KEYWORDS = [
    "alias",
    "and",
    "as",
    "assert",
    "break",
    "capturing",
    "class",
    "comptime",
    "continue",
    "def",
    "deinit",
    "del",
    "elif",
    "else",
    "escaping",
    "except",
    "exec",
    "finally",
    "for",
    "from",
    "fn",  # Used to be a keyword
    "global",
    "__generator_type",
    "if",
    "imm",
    "import",
    "in",
    "is",
    "lambda",
    "mut",
    "nonlocal",
    "not",
    "or",
    "out",
    "owned",  # Used to be a keyword
    "pass",
    "print",
    "raise",
    "raises",
    "read", # Used to be a keyword
    "ref",
    "return",
    "struct",
    "trait",
    "try",
    "unified",  # Used to be a keyword
    "var",
    "where",
    "while",
    "with",
    "yield",
]


@pytest.mark.parametrize("kw", KEYWORDS)
def test_keyword_as_method_name(kw):
    """Keywords can be used as a member method name."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        f"    def {kw}(self): pass\n"
        "def main():\n"
        "    var x = Foo()\n"
        f"    x.{kw}()\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        f"    def {kw}(self):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        f"    x.{kw}()\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", KEYWORDS)
def test_keyword_as_struct_name(kw):
    """Keywords can be used as a struct name, with backticks."""
    source = (
        "@fieldwise_init\n"
        f"struct `{kw}`: pass\n"
        "def main():\n"
        f"    var _ = `{kw}`()\n"
    )
    expected = (
        "@fieldwise_init\n"
        f"struct `{kw}`:\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        f"    var _ = `{kw}`()\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", KEYWORDS)
def test_keyword_as_trait_name(kw):
    """Keywords can be used as a trait name, with backticks."""
    source = (
        f"trait `{kw}`:\n"
        f"    def {kw}(self): pass\n"
        "\n"
        f"struct Foo(`{kw}`):\n"
        f"    def {kw}(self): pass\n"
    )
    expected = (
        f"trait `{kw}`:\n"
        f"    def {kw}(self):\n"
        "        pass\n"
        "\n"
        "\n"
        f"struct Foo(`{kw}`):\n"
        f"    def {kw}(self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("kw", KEYWORDS)
def test_keyword_as_mlir_region_name(kw):
    """Keywords can be used as __mlir_region names (with backticks)."""
    source = (
        "def foo():\n"
        f"    __mlir_region `{kw}`(): __mlir_op.`co.suspend.end`()\n"
        f'    __mlir_op.`co.suspend`[_region="{kw}".value]()\n'
    )
    expected = (
        "def foo():\n"
        f"    __mlir_region `{kw}`():\n"
        "        __mlir_op.`co.suspend.end`()\n"
        "\n"
        f'    __mlir_op.`co.suspend`[_region="{kw}".value]()\n'
    )
    assert_mojo_format(source, expected)
