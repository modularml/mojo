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

# Test that mojo format handles valid code where the colon is on the next line
# after a compound statement header.

import pytest

from tests.util import assert_mojo_format


def test_func_colon_next_line():
    """Function with colon on the next line."""
    source = "def foo()\n    :\n    pass\n"
    expected = "def foo():\n    pass\n"
    assert_mojo_format(source, expected)


def test_func_return_type_colon_next_line():
    """Function with return type and colon on the next line."""
    source = "def foo() -> Int\n    :\n    return 0\n"
    expected = "def foo() -> Int:\n    return 0\n"
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("keyword", ["struct", "trait"])
def test_type_decl_colon_next_line(keyword):
    """Struct/trait with colon on the next line."""
    source = f"{keyword} Foo\n    :\n    pass\n"
    expected = f"{keyword} Foo:\n    pass\n"
    assert_mojo_format(source, expected)


def test_if_colon_next_line():
    """if statement with colon on the next line."""
    source = "def main():\n    if True\n        :\n        pass\n"
    expected = "def main():\n    if True:\n        pass\n"
    assert_mojo_format(source, expected)


def test_for_colon_next_line():
    """for loop with colon on the next line."""
    source = "def main():\n    for i in range(1)\n        :\n        pass\n"
    expected = "def main():\n    for i in range(1):\n        pass\n"
    assert_mojo_format(source, expected)


def test_while_colon_next_line():
    """while loop with colon on the next line."""
    source = "def main():\n    while False\n        :\n        pass\n"
    expected = "def main():\n    while False:\n        pass\n"
    assert_mojo_format(source, expected)


def test_nested_colon_next_line():
    """Colon on next line inside an already-indented block."""
    source = (
        "struct Foo:\n"
        "    def bar(self)\n"
        "        :\n"
        "        pass\n"
    )
    expected = (
        "struct Foo:\n"
        "    def bar(self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiline_signature_colon_next_line():
    """Function with multi-line signature and colon on the next line."""
    source = (
        "def foo(\n"
        "    x: Int,\n"
        "    y: Int,\n"
        ")\n"
        "    :\n"
        "    pass\n"
    )
    expected = (
        "def foo(\n"
        "    x: Int,\n"
        "    y: Int,\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_trailing_comment_colon_next_line():
    """Colon on next line when the previous line has a trailing comment."""
    source = (
        "def foo()  # comment\n"
        "    :\n"
        "    pass\n"
    )
    expected = (
        "def foo():  # comment\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_colon_next_line_fmt_off():
    """A colon on the next line inside a fmt: off block should not be joined.

    Tests ``_normalize_mojo_source`` directly because the un-normalized
    colon-on-next-line syntax crashes the lib2to3 parser.  If the parser
    learns to handle this syntax natively, this test can be changed to
    use ``assert_mojo_format`` instead.
    """
    from mblib2to3.pgen2.driver import _normalize_mojo_source

    source = (
        "# fmt: off\n"
        "def foo()\n"
        "    :\n"
        "    pass\n"
        "# fmt: on\n"
    )
    assert _normalize_mojo_source(source) == source


def test_colon_in_docstring_preserved():
    """A colon-only line inside a docstring must not be altered."""
    source = (
        'def main()\n'
        '    :\n'
        '    """\n'
        '    :\n'
        '    """\n'
        '    pass\n'
    )
    expected = (
        'def main():\n'
        '    """\n'
        '    :\n'
        '    """\n'
        '    pass\n'
    )
    assert_mojo_format(source, expected)
