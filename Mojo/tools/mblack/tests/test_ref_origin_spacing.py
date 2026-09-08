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


def test_ref_origin_in_parameter():
    """No space between ref and [origin] in parameters."""
    source = "def foo(ref[MutAnyOrigin] r: Int): pass"
    expected = (
        "def foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_parameter_with_space():
    """Space between ref and [origin] should be removed."""
    source = "def foo(ref [MutAnyOrigin] r: Int): pass"
    expected = (
        "def foo(ref[MutAnyOrigin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type():
    """No space between ref and [origin] in return types."""
    source = "def bar(x: String) raises Int -> ref[x] String: return x"
    expected = (
        "def bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_in_return_type_with_space():
    """Space between ref and [origin] in return types should be removed."""
    source = "def bar(x: String) raises Int -> ref [x] String: return x"
    expected = (
        "def bar(x: String) raises Int -> ref[x] String:\n"
        "    return x\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_multiple_origins():
    """Multiple origins should be formatted correctly."""
    source = "def foo[a: Origin, b: Origin](ref[a, b] r: Int): pass"
    expected = (
        "def foo[a: Origin, b: Origin](ref[a, b] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_with_space_before_param():
    """Ensure space is preserved between ] and parameter name."""
    source = "def foo[origin: Origin](ref[origin]r: Int): pass"
    expected = (
        "def foo[origin: Origin](ref[origin] r: Int):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param():
    """Space after ref[origin] with untyped parameter (e.g., self)."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED]self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_space():
    """Preserve correct spacing for untyped parameter."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_ref_origin_untyped_param_with_extra_spaces():
    """Fix extra space between ref and [origin] for untyped parameter."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref [AddressSpace.SHARED] self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def barriers(ref[AddressSpace.SHARED] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


@pytest.mark.parametrize("space", ["", " "])
def test_out_address_space_in_initializer(space):
    """Any space between out and [address_space] is removed in initializers."""
    source = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        f"    def __init__(out{space}[AddressSpace.GENERIC] self): pass\n"
    )
    expected = (
        "from std.memory.pointer import AddressSpace\n"
        "\n"
        "\n"
        "struct Foo:\n"
        "    def __init__(out[AddressSpace.GENERIC] self):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)
