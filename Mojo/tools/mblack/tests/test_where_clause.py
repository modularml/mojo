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

from tests.util import assert_mojo_format

# ====================== #
# Fns with where clauses
# ====================== #

def test_simple_fn_where_clause():
    source = "def where_simple[x: Bool]() where x: pass"
    expected = (
        "def where_simple[x: Bool]() where x:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_composite_fn_where_clause_with_return():
    source = (
        "def where_composite[x: Bool, y: Int, z: Int]() -> Int"
        " where x and y + z > 0: return 0"
    )
    expected = (
        "def where_composite[x: Bool, y: Int, z: Int]() -> Int"
        " where x and y + z > 0:\n"
        "    return 0\n"
    )
    assert_mojo_format(source, expected)


def test_long_fn_where_clause():
    source = (
        "def where_long[x: Bool, y: Int, z: Int, a: Int, b: Int, c: Int,"
        " d: Int, e: Int, f: Int, g: Int](\n"
        "    p: Int, q: Int, r: Int\n"
        ") where x and   y + z > 0 and a == b and c   == d and"
        " (e == f or  g > 0):\n"
        "    pass\n"
    )
    expected = (
        "def where_long[\n"
        "    x: Bool,\n"
        "    y: Int,\n"
        "    z: Int,\n"
        "    a: Int,\n"
        "    b: Int,\n"
        "    c: Int,\n"
        "    d: Int,\n"
        "    e: Int,\n"
        "    f: Int,\n"
        "    g: Int,\n"
        "](p: Int, q: Int, r: Int) where (\n"
        "    x and y + z > 0 and a == b and c == d and (e == f or g > 0)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_very_long_fn_where_clause_with_return():
    source = (
        "def where_very_long[x: Bool, y: Int, z: Int, a: Int, b: Int,"
        " c: Int, d: Int, e: Int, f: Int, g: Int, xx: Int, yy: Int,"
        " zz: Int, aa: Int, ds: Bool, dd: Bool]()"
        " -> Tuple[Int, Int, String] where (\n"
        "    x and y + z > 0 and a == b and c == d and (e == f or g > 0)"
        " and xx == yy and zz == aa or (ds or dd)\n"
        "):\n"
        '    return (0, 0, "")\n'
    )
    expected = (
        "def where_very_long[\n"
        "    x: Bool,\n"
        "    y: Int,\n"
        "    z: Int,\n"
        "    a: Int,\n"
        "    b: Int,\n"
        "    c: Int,\n"
        "    d: Int,\n"
        "    e: Int,\n"
        "    f: Int,\n"
        "    g: Int,\n"
        "    xx: Int,\n"
        "    yy: Int,\n"
        "    zz: Int,\n"
        "    aa: Int,\n"
        "    ds: Bool,\n"
        "    dd: Bool,\n"
        "]() -> Tuple[Int, Int, String] where (\n"
        "    x\n"
        "    and y + z > 0\n"
        "    and a == b\n"
        "    and c == d\n"
        "    and (e == f or g > 0)\n"
        "    and xx == yy\n"
        "    and zz == aa\n"
        "    or (ds or dd)\n"
        "):\n"
        '    return (0, 0, "")\n'
    )
    assert_mojo_format(source, expected)


def test_multiple_fn_where_clauses():
    source = (
        "def where_multiple[x: Bool, y: Int, z: Int, a: Int, b: Int,"
        " c: Int, d: Int, e: Int, f: Int, g: Int, xx: Int, yy: Int,"
        " zz: Int, aa: Int, ds: Bool, dd: Bool]() "
        "where x and y + z > 0 and a == b "
        "where c == d and (e == f or g > 0) and xx == yy "
        "where zz == aa or (ds or dd):\n"
        "    pass\n"
    )
    expected = (
        "def where_multiple[\n"
        "    x: Bool,\n"
        "    y: Int,\n"
        "    z: Int,\n"
        "    a: Int,\n"
        "    b: Int,\n"
        "    c: Int,\n"
        "    d: Int,\n"
        "    e: Int,\n"
        "    f: Int,\n"
        "    g: Int,\n"
        "    xx: Int,\n"
        "    yy: Int,\n"
        "    zz: Int,\n"
        "    aa: Int,\n"
        "    ds: Bool,\n"
        "    dd: Bool,\n"
        "]() where x and y + z > 0 and a == b where (\n"
        "    c == d and (e == f or g > 0) and xx == yy\n"
        ") where zz == aa or (ds or dd):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


# ============================ #
# Structs with where clauses
# ============================ #


def test_simple_struct_where_clause():
    source = "struct WhereSimple[T: Bool] where T: pass"
    expected = (
        "struct WhereSimple[T: Bool] where T:\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_struct_where_clause_no_params():
    """A bare `where` clause with no parameter list and no parent traits."""
    source = (
        "def some_predicate() -> Bool:\n"
        "    return True\n"
        "\n"
        "\n"
        "struct NoParams where some_predicate(): pass\n"
    )
    expected = (
        "def some_predicate() -> Bool:\n"
        "    return True\n"
        "\n"
        "\n"
        "struct NoParams where some_predicate():\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_long_struct_where_clause():
    source = (
        "struct WhereLong[T: Bool, U: Int, V: Int, W: Int, a: Int, b: Int,"
        " c: Int, d: Int, e: Int, f: Int, g: Int]() "
        "where T and U + V == W and a == b and c == d and"
        " (e == f or g > 0): pass"
    )
    expected = (
        "struct WhereLong[\n"
        "    T: Bool,\n"
        "    U: Int,\n"
        "    V: Int,\n"
        "    W: Int,\n"
        "    a: Int,\n"
        "    b: Int,\n"
        "    c: Int,\n"
        "    d: Int,\n"
        "    e: Int,\n"
        "    f: Int,\n"
        "    g: Int,\n"
        "]() where (\n"
        "    T and U + V == W and a == b and c == d and (e == f or g > 0)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_very_long_struct_where_clause():
    source = (
        "struct WhereVeryLong[x: Bool, y: Int, z: Int, a: Int, b: Int,"
        " c: Int, d: Int, e: Int, f: Int, g: Int, xx: Int, yy: Int,"
        " zz: Int, aa: Int, ds: Bool, dd: Bool] where (\n"
        "    x and y + z > 0 and a == b and c == d and (e == f or g > 0)"
        " and xx == yy and zz == aa or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    expected = (
        "struct WhereVeryLong[\n"
        "    x: Bool,\n"
        "    y: Int,\n"
        "    z: Int,\n"
        "    a: Int,\n"
        "    b: Int,\n"
        "    c: Int,\n"
        "    d: Int,\n"
        "    e: Int,\n"
        "    f: Int,\n"
        "    g: Int,\n"
        "    xx: Int,\n"
        "    yy: Int,\n"
        "    zz: Int,\n"
        "    aa: Int,\n"
        "    ds: Bool,\n"
        "    dd: Bool,\n"
        "] where (\n"
        "    x\n"
        "    and y + z > 0\n"
        "    and a == b\n"
        "    and c == d\n"
        "    and (e == f or g > 0)\n"
        "    and xx == yy\n"
        "    and zz == aa\n"
        "    or (ds or dd)\n"
        "):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_multiple_struct_where_clauses_no_parent():
    source = (
        "struct WhereMultiple[T: Movable] "
        "where conforms_to(T, Copyable) where conforms_to(T, Intable): pass"
    )
    expected = (
        "struct WhereMultiple[T: Movable] where conforms_to(\n"
        "    T, Copyable\n"
        ") where conforms_to(T, Intable):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)


def test_trailing_where_clause_on_struct_decl():
    source = (
        "struct Wrapper[\n"
        "T: Movable, U: Movable\n"
        "](\n"
        "Movable\n"
        ") where conforms_to(T, Copyable) and conforms_to(U, Intable): pass"
    )
    expected = (
        "struct Wrapper[T: Movable, U: Movable](Movable) where conforms_to(\n"
        "    T, Copyable\n"
        ") and conforms_to(U, Intable):\n"
        "    pass\n"
    )
    assert_mojo_format(source, expected)
