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

# Tests that a line continuation is rejoined when it follows an operand that
# itself spans multiple lines inside brackets.
#
# The line normalizer compares a continuation's indent against the indent of
# the current logical statement's *head*.

from tests.util import assert_mojo_format


def test_ternary_after_multiline_call_operand():
    """A ternary whose first operand is a multi-line call rejoins."""
    source = (
        "def twice(n: Int) -> Int:\n"
        "    return n * 2\n"
        "\n"
        "\n"
        "def choose(cond: Bool) -> Int:\n"
        "    var x =\n"
        "        twice(\n"
        "            5\n"
        "        )\n"
        "        if cond\n"
        "        else 3\n"
        "    return x\n"
    )
    expected = (
        "def twice(n: Int) -> Int:\n"
        "    return n * 2\n"
        "\n"
        "\n"
        "def choose(cond: Bool) -> Int:\n"
        "    var x = twice(5) if cond else 3\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_trailing_operator_after_multiline_call_operand():
    """A binary-operator continuation after a multi-line call operand rejoins."""
    source = (
        "def twice(n: Int) -> Int:\n"
        "    return n * 2\n"
        "\n"
        "\n"
        "def add_one() -> Int:\n"
        "    var x =\n"
        "        twice(\n"
        "            5\n"
        "        ) +\n"
        "        1\n"
        "    return x\n"
    )
    expected = (
        "def twice(n: Int) -> Int:\n"
        "    return n * 2\n"
        "\n"
        "\n"
        "def add_one() -> Int:\n"
        "    var x = twice(5) + 1\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_dot_chain_after_multiline_call_operand():
    """A dot-method continuation after a multi-line call operand rejoins."""
    source = (
        "@fieldwise_init\n"
        "struct Box(Copyable, Movable):\n"
        "    var v: Int\n"
        "\n"
        "    def doubled(self) -> Int:\n"
        "        return self.v * 2\n"
        "\n"
        "\n"
        "def use_box() -> Int:\n"
        "    var x =\n"
        "        Box(\n"
        "            5\n"
        "        )\n"
        "        .doubled()\n"
        "    return x\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Box(Copyable, Movable):\n"
        "    var v: Int\n"
        "\n"
        "    def doubled(self) -> Int:\n"
        "        return self.v * 2\n"
        "\n"
        "\n"
        "def use_box() -> Int:\n"
        "    var x = Box(5).doubled()\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)
