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

# Tests for dot-prefixed method chain continuation across lines.

from tests.util import assert_mojo_format


def test_dot_continuation():
    """A dot-prefixed method call on a continuation line should be joined."""
    source = (
        "def main():\n"
        '    var text = String("hello")\n'
        "           .upper()\n"
    )
    expected = (
        "def main():\n"
        '    var text = String("hello").upper()\n'
    )
    assert_mojo_format(source, expected)


def test_dot_continuation_with_backslash():
    """Backslash-continued dot method call should also work."""
    source = (
        "def main():\n"
        '    var text = String("hello") \\\n'
        "           .upper()\n"
    )
    expected = (
        "def main():\n"
        '    var text = String("hello").upper()\n'
    )
    assert_mojo_format(source, expected)


def test_dot_continuation_with_comment():
    """A trailing comment on the previous line should be preserved."""
    source = (
        "def main():\n"
        '    var text = String("hello")  # create string\n'
        "           .upper()\n"
    )
    expected = (
        "def main():\n"
        '    var text = String("hello").upper()  # create string\n'
    )
    assert_mojo_format(source, expected)


def test_dot_continuation_chained():
    """Multiple dot-continuations should all be joined."""
    source = (
        "def main():\n"
        '    var text = String("hello")\n'
        "           .upper()\n"
        "           .lower()\n"
    )
    expected = (
        "def main():\n"
        '    var text = String("hello").upper().lower()\n'
    )
    assert_mojo_format(source, expected)


def test_dot_at_same_indent_not_joined():
    """A dot line at same or lesser indent should not be joined — it's a separate
    statement (e.g. an inferred member reference), not a continuation."""
    source = (
        "def main():\n"
        '    var x = String("hello")\n'
        "    .upper()\n"
    )
    assert_mojo_format(source, source)


def test_ellipsis_not_joined():
    """An ellipsis (``...``) on its own line should not be joined."""
    source = (
        "struct Foo:\n"
        '    """Docstring."""\n'
        "\n"
        "    ...\n"
    )
    assert_mojo_format(source, source)


def test_dot_continuation_fmt_off():
    """A dot continuation inside a fmt: off block should not be joined.

    Tests ``_normalize_mojo_source`` directly because the un-normalized
    dot continuation crashes the lib2to3 parser.
    """
    from mblib2to3.pgen2.driver import _normalize_mojo_source

    source = (
        "def main():\n"
        "    # fmt: off\n"
        '    var text = String("hello")\n'
        "           .upper()\n"
        "    # fmt: on\n"
    )
    assert _normalize_mojo_source(source) == source
