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

# Tests that the line normalizer's code-bracket scan treats Mojo backtick
# strings (raw identifiers / MLIR attributes, e.g. ``__mlir_attr.`[#llvm.x]```)
# as opaque. A ``#`` inside a backtick must not be read as a comment, and
# brackets inside it must not be counted -- otherwise ``bracket_depth`` desyncs
# and line rejoining is silently disabled for the rest of the scope, so a later
# continuation fails to parse ("Cannot parse").
#
# The samples use ``__mlir_attr`` constructs that do not ``mojo build`` as
# standalone units (the attributes need their dialect), so this file is excluded
# from ``--validate-with-mojo-build`` (see BUILD.bazel). The formatting behavior
# is what is under test.

from tests.util import assert_mojo_format


def test_backtick_mlir_attr_with_hash_does_not_desync_continuation():
    """A ``#`` inside a backtick must not disable rejoining the next statement."""
    source = (
        "def main():\n"
        "    comptime a = __mlir_attr.`[#llvm.foo]`\n"
        "    comptime x =\n"
        "        1 + 2\n"
    )
    expected = (
        "def main():\n"
        "    comptime a = __mlir_attr.`[#llvm.foo]`\n"
        "    comptime x = 1 + 2\n"
    )
    assert_mojo_format(source, expected)


def test_backtick_mlir_attr_with_brackets_and_string_does_not_desync():
    """Real-world shape: a backtick attribute containing brackets, ``#`` and a
    quoted string, followed by a continuation that must still rejoin."""
    source = (
        "def main():\n"
        '    comptime s = __mlir_attr.`[#llvm.alias_scope<id="a">]`\n'
        "    comptime y =\n"
        "        3 + 4\n"
    )
    expected = (
        "def main():\n"
        '    comptime s = __mlir_attr.`[#llvm.alias_scope<id="a">]`\n'
        "    comptime y = 3 + 4\n"
    )
    assert_mojo_format(source, expected)

def test_backtick_before_trailing_operator_still_rejoins():
    """A ``#`` inside a backtick must not break the trailing-operator join.

    Joining a trailing ``+`` first strips the previous line's comment; if the
    ``#`` inside the backtick is read as a comment, the ``+`` is dropped from
    the code and the continuation never rejoins.
    """
    source = (
        "def main():\n"
        "    comptime a = __mlir_attr.`[#llvm.foo]` +\n"
        "        b\n"
    )
    expected = (
        "def main():\n"
        "    comptime a = __mlir_attr.`[#llvm.foo]` + b\n"
    )
    assert_mojo_format(source, expected)


def test_backtick_inside_tstring_interpolation_is_opaque():
    """A backtick inside an t-string interpolation must be skipped whole.

    A quote inside the backtick must not be read as a nested string (cf. the
    ``t"{"("}"`` case): doing so desyncs the interpolation and leaves the
    line's bracket depth wrong, disabling rejoining for the rest of the scope.
    """
    source = (
        "def main():\n"
        '    x = String(t"{a.`"`}")\n'
        "    y =\n"
        "        1\n"
    )
    expected = (
        "def main():\n"
        '    x = String(t"{a.`"`}")\n'
        "    y = 1\n"
    )
    assert_mojo_format(source, expected)
