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

# Tests that mblack does not insert a space between `*` and a complex
# expression (e.g. `Self.element_types`) in variadic pack unpacking
# annotations.

from tests.util import assert_mojo_format


def test_star_simple_name_in_param():
    """`*Ts` (simple name) in a parameter annotation stays unchanged."""
    source = "def f[*Ts: Movable](*args: * Ts):\n    pass\n"
    expected = "def f[*Ts: Movable](*args: *Ts):\n    pass\n"
    assert_mojo_format(source, expected)


def test_star_self_attr_in_param():
    """A stray space between `*` and `Self.X` is removed."""
    source = (
        "struct Tuple[*Ts: Movable]:\n"
        "    def __init__(out self, var *args: * Self.Ts):\n"
        "        pass\n"
    )
    expected = (
        "struct Tuple[*Ts: Movable]:\n"
        "    def __init__(out self, var *args: *Self.Ts):\n"
        "        pass\n"
    )
    assert_mojo_format(source, expected)


def test_star_expr_with_trailers_in_subscript():
    """`*Name.attr[T, U]()` in a subscript: no space after `*` regardless of
    how deep the trailer chain goes."""
    source = (
        "struct Holder[*Ts: Movable]:\n"
        "    pass\n"
        "\n"
        "\n"
        "comptime MyHolder = Holder[*TypeList.of[Int, Int]()]\n"
    )
    assert_mojo_format(source, source)


def test_star_subscripted_name_in_param():
    """`*SomeTypeList[Writable]` in a parameter annotation: no space."""
    source = "def foo(*args: * SomeTypeList[Writable]):\n    pass\n"
    expected = "def foo(*args: *SomeTypeList[Writable]):\n    pass\n"
    assert_mojo_format(source, expected)


def test_star_param_followed_by_other_param():
    """`*args: *Ts` followed by another parameter: no space after `*`."""
    source = "def f[*Ts: Movable](*args: *Ts, other: Int):\n    pass\n"
    assert_mojo_format(source, source)
