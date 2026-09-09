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

# Tests for consistent spacing of keyword parameter bindings in square brackets,
# and test for similar patterns in function calls etc to ensure no regressions.

from tests.util import assert_mojo_format


def test_call_params_literals():
    """Spaces around = with literal values should be removed."""
    source = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a = 1, b = 0]()\n"
    )
    expected = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a=1, b=0]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_function():
    """Spaces around = in parameter bindings should be removed."""
    source = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a = g(), b = 0]()\n"
    )
    expected = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a=g(), b=0]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_two_functions():
    """Spaces around = in parameter bindings should be removed."""
    source = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a = g(), b = g()]()\n"
    )
    expected = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a=g(), b=g()]()\n"
    )
    assert_mojo_format(source, expected)


def test_type_params_literals():
    """Spaces around = in type parameter bindings should be removed."""
    source = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x: Foo[a = 1, b = 0]\n"
    )
    expected = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x: Foo[a=1, b=0]\n"
    )
    assert_mojo_format(source, expected)


def test_type_params_function():
    """Spaces around = in type parameter bindings should be removed."""
    source = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x: Foo[a = g(), b = 0]\n"
    )
    expected = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x: Foo[a=g(), b=0]\n"
    )
    assert_mojo_format(source, expected)


def test_alias_params_literals():
    """Spaces around = in alias parameter bindings should be removed."""
    source = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "comptime MyFoo = Foo[a = 1, b = 0]\n"
    )
    expected = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "comptime MyFoo = Foo[a=1, b=0]\n"
    )
    assert_mojo_format(source, expected)


def test_alias_params_function():
    """Spaces around = in alias parameter bindings should be removed."""
    source = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "comptime MyFoo = Foo[a = g(), b = 0]\n"
    )
    expected = (
        "struct Foo[a: Int, b: Int]:\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "comptime MyFoo = Foo[a=g(), b=0]\n"
    )
    assert_mojo_format(source, expected)


# A small Foo with an attribute and a method, used by tests whose samples
# exercise attribute access / method call as a parameter binding value.
_FOO_DECL = (
    "@fieldwise_init\n"
    "struct Foo(Copyable):\n"
    "    var y: Int\n"
    "    var _y: Int\n"
    "\n"
    "    def method(self) -> Int:\n"
    "        return self.y\n"
    "\n"
    "\n"
)


def test_call_params_attribute_access():
    """Spaces around = should be removed when value is an attribute access."""
    source = (
        _FOO_DECL
        + "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a = x.y]()\n"
        "    f[a = x._y]()\n"
    )
    expected = (
        _FOO_DECL
        + "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a=x.y]()\n"
        "    f[a=x._y]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_method_call():
    """Spaces around = should be removed when value is a method call."""
    source = (
        _FOO_DECL
        + "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a = x.method()]()\n"
    )
    expected = (
        _FOO_DECL
        + "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a=x.method()]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_arithmetic():
    """Spaces around = should be removed when value is arithmetic."""
    source = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a = 1 + 2, b = 3 * 4]()\n"
    )
    expected = (
        "def f[a: Int, b: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a=1 + 2, b=3 * 4]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_negation():
    """Spaces around = should be removed when value is a negation."""
    source = (
        "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a = -1]()\n"
    )
    expected = (
        "def f[a: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[a=-1]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_parameterized_type():
    """Spaces around = should be removed when value is a parameterized type."""
    source = (
        "def f[T: AnyType]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[T = List[Int]]()\n"
    )
    expected = (
        "def f[T: AnyType]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    f[T=List[Int]]()\n"
    )
    assert_mojo_format(source, expected)


def test_call_params_mixed_complex_and_simple():
    """Spaces around = should be consistently removed across mixed values."""
    source = (
        _FOO_DECL
        + "def f[a: Int, b: Int, c: Int, d: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a = g(), b = x.y, c = 1, d = 1 + 2]()\n"
    )
    expected = (
        _FOO_DECL
        + "def f[a: Int, b: Int, c: Int, d: Int]():\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    comptime x = Foo(0, 0)\n"
        "    f[a=g(), b=x.y, c=1, d=1 + 2]()\n"
    )
    assert_mojo_format(source, expected)


def test_mixed_param_binding_and_slice():
    """Parameter bindings get no spaces while slices in the same list keep them."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, a: Int, b: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[a = 1, b = 0:5]\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, a: Int, b: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[a=1, b=0:5]\n"
    )
    assert_mojo_format(source, expected)


def test_func_args():
    """Spaces around = in function argument bindings should be removed."""
    source = (
        "def f(a: Int, b: Int):\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f(a = g(), b = 0)\n"
    )
    expected = (
        "def f(a: Int, b: Int):\n"
        "    pass\n"
        "\n"
        "\n"
        "def g() -> Int:\n"
        "    return 42\n"
        "\n"
        "\n"
        "def main():\n"
        "    f(a=g(), b=0)\n"
    )
    assert_mojo_format(source, expected)
