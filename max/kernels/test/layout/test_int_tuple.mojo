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

from layout.coord import Coord, Idx
from layout.int_tuple import *
from layout.int_tuple import abs  # override builtin abs and min
from std.testing import assert_equal, assert_false, assert_true, assert_raises


def test_tuple_basic() raises:
    print("== test_tuple_basic")

    # Test len() operator
    assert_equal(len(IntTuple()), 0)
    assert_equal(len(IntTuple(1)), 1)
    assert_equal(len(IntTuple(1, 2)), 2)
    assert_equal(len(IntTuple(1, IntTuple(2, 3))), 2)

    # Test single integer value tuple
    comptime t0: IntTuple = 5
    assert_equal(String(t0), "5")

    # Test simple tuple compositions
    assert_equal(String(IntTuple(IntTuple(IntTuple()))), "((()))")
    assert_equal(String(IntTuple(IntTuple(IntTuple(3)))), "((3))")
    assert_equal(
        String(IntTuple(7, IntTuple(2, 3, 4, IntTuple(5, 6)))),
        "(7, (2, 3, 4, (5, 6)))",
    )
    assert_equal(String(IntTuple(2, IntTuple(3, 4))), "(2, (3, 4))")

    # Test basic tuple operations
    var tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )
    assert_equal(String(tt), "(5, 7, 2, (3, 66, (6, 99, (4, 68, 721))), 42)")

    # tt[1] = 8
    # tt.append(81)
    # assert_equal(String(tt), "(5, 8, 2, (3, 66, (6, 99, (4, 68, 721))), 42, 81)")

    # tt[3][2][2] = IntTuple(5, 69, 722)
    # assert_equal(String(tt), "(5, 8, 2, (3, 66, (6, 99, (5, 69, 722))), 42, 81)")

    # Tests interaction with compiler interpreter
    comptime works = IntTuple(IntTuple(2, 2), IntTuple(2, 3))
    assert_equal(String(works), "((2, 2), (2, 3))")

    comptime works_too = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    assert_equal(String(works_too), "((2, 2), (2, 2))")

    # Tests IntTuple equality operations
    assert_equal(IntTuple(1, 2) == IntTuple(1, 2), True)
    assert_equal(IntTuple(1, 2) == IntTuple(1, 3), False)
    assert_equal(IntTuple(1, 2) == IntTuple(1, 2, 3), False)
    assert_equal(
        IntTuple(1, 2, IntTuple(3, 4)) == IntTuple(1, 2, IntTuple(3, 4)), True
    )
    assert_equal(
        IntTuple(1, 2, IntTuple(2, 4)) == IntTuple(1, 2, IntTuple(3, 4)), False
    )
    assert_equal(
        IntTuple(1, 2, IntTuple(3, 5)) == IntTuple(1, 2, IntTuple(3, 4)), False
    )


def test_tuple_slicing() raises:
    print("== test_tuple_slicing")

    comptime tr = IntTuple(0, 1, 2, 3, 4)
    comptime sl0 = String(tr[len(tr) - 1])
    comptime sl1 = String(tr[len(tr) - 2])
    comptime sl2 = String(tr[1:4])
    comptime sl3 = String(tr[1:5:2])
    comptime sl4 = String(tr[:])
    comptime sl5 = String(tr[:5:2])
    comptime sl6 = String(tr[-3:])
    comptime sl7 = String(tr[-3:-1])
    comptime sl8 = String(tr[:-1])
    assert_equal(sl0, "4")
    assert_equal(sl1, "3")
    assert_equal(sl2, "(1, 2, 3)")
    assert_equal(sl3, "(1, 3)")
    assert_equal(sl4, "(0, 1, 2, 3, 4)")
    assert_equal(sl5, "(0, 2, 4)")
    assert_equal(sl6, "(2, 3, 4)")
    assert_equal(sl7, "(2, 3)")
    assert_equal(sl8, "(0, 1, 2, 3)")


def test_tuple_basic_ops() raises:
    print("== test_tuple_basic_ops")

    comptime p0 = product(2)
    comptime p1 = product(IntTuple(3, 2))
    comptime p2 = product(IntTuple(IntTuple(2, 3), 4))
    comptime p3 = product([[2, 3], 4])
    assert_equal(String(p0), "2")
    assert_equal(String(p1), "6")
    assert_equal(String(p2), "24")

    comptime tt = IntTuple(
        5,
        7,
        2,
        IntTuple(3, 66, IntTuple(6, 99, IntTuple(4, 68, 721))),
        42,
    )

    comptime f = flatten(tt)
    assert_equal(String(f), "(5, 7, 2, 3, 66, 6, 99, 4, 68, 721, 42)")

    comptime tt_unknown = to_unknown(tt)
    assert_equal(
        String(tt_unknown),
        "(-1, -1, -1, (-1, -1, (-1, -1, (-1, -1, -1))), -1)",
    )

    comptime ts = IntTuple(0, 1, IntTuple(-2, 3), -4)
    assert_equal(String(abs(ts)), "(0, 1, (2, 3), 4)")

    comptime tm = IntTuple(0, 1, IntTuple(2, 3), 4)
    assert_equal(String(mul(tm, 4)), "(0, 4, (8, 12), 16)")

    comptime s = sum(IntTuple(IntTuple(2, 3), 4))
    assert_equal(s, 9)

    comptime ip1 = inner_product(IntTuple(2), IntTuple(3))
    comptime ip2 = inner_product(IntTuple(1, 2), IntTuple(3, 2))
    comptime ip3 = inner_product(
        IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2)
    )
    assert_equal(ip1, 6)
    assert_equal(ip2, 7)
    assert_equal(ip3, 15)

    comptime m0 = tuple_max(
        IntTuple(1, 2, 3, IntTuple(4, 5), IntTuple(7, 8, 9, 10))
    )
    assert_equal(m0, 10)

    assert_equal(
        tuple_min(IntTuple(1, 5, 6), IntTuple(4, 2, 3)), IntTuple(1, 2, 3)
    )

    assert_equal(
        tuple_min(
            IntTuple(1, IntTuple(14, 6)),
            IntTuple(4, IntTuple(2, 32)),
        ),
        IntTuple(1, IntTuple(2, 6)),
    )


def test_sorted() raises:
    print("== test_sorted")

    comptime t0 = sorted(IntTuple(7, 3, 1, 5, 0))
    assert_equal(String(t0), "(0, 1, 3, 5, 7)")

    comptime t1 = sorted(IntTuple(IntTuple(7, 3), IntTuple(1, 5, 0)))
    assert_equal(String(t1), "((1, 5, 0), (7, 3))")

    comptime t2 = sorted(IntTuple(IntTuple(7, 3), IntTuple(1, IntTuple(5, 0))))
    assert_equal(String(t2), "((1, (5, 0)), (7, 3))")

    assert_true(IntTuple(4, 6, 8) < IntTuple(5, 6, 7))


def test_apply() raises:
    print("== test_apply")

    def double(x: Int) {} -> Int:
        return x * 2

    assert_equal(apply(IntTuple(1, 2, 3), double), IntTuple(2, 4, 6))
    assert_equal(
        apply(IntTuple(1, IntTuple(2, 3), 4), double),
        IntTuple(2, IntTuple(4, 6), 8),
    )

    var offset = 10

    def add_offset(x: Int) {imm} -> Int:
        return x + offset

    assert_equal(
        apply(IntTuple(1, IntTuple(2, 3)), add_offset),
        IntTuple(11, IntTuple(12, 13)),
    )


def test_product() raises:
    print("== test_product")

    assert_equal(product(2), 2)
    assert_equal(product(IntTuple(3, 2)), 6)
    assert_equal(product(product(IntTuple(IntTuple(2, 3), 4))), 24)


def test_inner_product() raises:
    print("== test_inner_product")

    assert_equal(inner_product(2, 3), 6)
    assert_equal(inner_product(IntTuple(1, 2), IntTuple(3, 2)), 7)
    assert_equal(
        inner_product(IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(2, 1), 2)),
        15,
    )


def test_shape_div() raises:
    print("== test_shape_div")

    assert_equal(shape_div(IntTuple(3, 4), 6), IntTuple(1, 2))
    assert_equal(shape_div(IntTuple(3, 4), 12), IntTuple(1, 1))
    assert_equal(shape_div(IntTuple(3, 4), 36), IntTuple(1, 1))
    assert_equal(
        shape_div(IntTuple(IntTuple(3, 4), 6), 36), IntTuple(IntTuple(1, 1), 2)
    )
    assert_equal(
        shape_div(IntTuple(6, IntTuple(3, 4)), 36), IntTuple(1, IntTuple(1, 2))
    )


def test_prefix_product() raises:
    print("== test_prefix_product")

    assert_equal(prefix_product(2), 1)

    assert_equal(prefix_product(IntTuple(3, 2)), IntTuple(1, 3))

    assert_equal(prefix_product(IntTuple(3, 2, 4)), IntTuple(1, 3, 6))

    assert_equal(
        prefix_product(IntTuple(IntTuple(2, 3), 4)), IntTuple(IntTuple(1, 2), 6)
    )

    assert_equal(
        prefix_product(
            IntTuple(IntTuple(2, 3), IntTuple(2, 1, 2), IntTuple(5, 2, 1))
        ),
        IntTuple(IntTuple(1, 2), IntTuple(6, 12, 12), IntTuple(24, 120, 240)),
    )


def test_crd2idx() raises:
    print("== test_crd2idx")

    comptime cx0 = crd2idx(IntTuple(0, 0), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx1 = crd2idx(IntTuple(1, 0), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx2 = crd2idx(IntTuple(2, 0), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx3 = crd2idx(IntTuple(3, 0), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx4 = crd2idx(IntTuple(0, 1), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx5 = crd2idx(IntTuple(1, 1), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx6 = crd2idx(IntTuple(2, 1), IntTuple(4, 2), IntTuple(1, 4))
    comptime cx7 = crd2idx(IntTuple(3, 1), IntTuple(4, 2), IntTuple(1, 4))
    assert_equal(cx0, 0)
    assert_equal(cx1, 1)
    assert_equal(cx2, 2)
    assert_equal(cx3, 3)
    assert_equal(cx4, 4)
    assert_equal(cx5, 5)
    assert_equal(cx6, 6)
    assert_equal(cx7, 7)


def test_idx2crd() raises:
    print("== test_idx2crd")

    comptime xc0 = idx2crd(0, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc1 = idx2crd(1, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc2 = idx2crd(2, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc3 = idx2crd(3, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc4 = idx2crd(4, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc5 = idx2crd(5, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc6 = idx2crd(6, IntTuple(4, 2), IntTuple(1, 4))
    comptime xc7 = idx2crd(7, IntTuple(4, 2), IntTuple(1, 4))
    assert_equal(String(xc0), "(0, 0)")
    assert_equal(String(xc1), "(1, 0)")
    assert_equal(String(xc2), "(2, 0)")
    assert_equal(String(xc3), "(3, 0)")
    assert_equal(String(xc4), "(0, 1)")
    assert_equal(String(xc5), "(1, 1)")
    assert_equal(String(xc6), "(2, 1)")
    assert_equal(String(xc7), "(3, 1)")


def test_weakly_congruent() raises:
    print("== test_weakly_congruent")
    comptime a = IntTuple(1)
    comptime b = IntTuple(2)

    assert_true(weakly_congruent(a, a))

    comptime a0 = IntTuple(IntTuple(1))
    comptime b0 = IntTuple(IntTuple(2))
    assert_true(weakly_congruent(a, a0))
    assert_true(weakly_congruent(b, b0))
    assert_true(weakly_congruent(a, b0))
    assert_true(weakly_congruent(b, a0))
    assert_false(weakly_congruent(a0, a))
    assert_false(weakly_congruent(b0, b))
    assert_false(weakly_congruent(a0, b))
    assert_false(weakly_congruent(b0, a))
    assert_true(weakly_congruent(a0, a0))
    assert_true(weakly_congruent(b0, b0))
    assert_true(weakly_congruent(a0, b0))

    comptime a1 = IntTuple(1, 1)
    assert_true(weakly_congruent(a, a1))
    assert_false(weakly_congruent(a0, a1))
    assert_true(weakly_congruent(a1, a1))

    comptime a2 = IntTuple(1, IntTuple(1, 1))
    assert_true(weakly_congruent(a, a2))
    assert_false(weakly_congruent(a0, a2))
    assert_true(weakly_congruent(a1, a2))

    comptime b1 = IntTuple(2, 2)
    assert_true(weakly_congruent(b, b1))
    assert_false(weakly_congruent(b0, b1))
    assert_true(weakly_congruent(a1, b1))

    comptime b2 = IntTuple(2, IntTuple(2, 2))
    assert_false(weakly_congruent(a2, b0))
    assert_false(weakly_congruent(a2, a1))
    assert_true(weakly_congruent(a2, b2))

    comptime b3 = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    assert_false(weakly_congruent(a0, b3))
    assert_true(weakly_congruent(a1, b3))
    assert_true(weakly_congruent(a2, b3))


def test_weakly_compatible() raises:
    print("== test_weakly_compatible")
    comptime a = IntTuple(16)
    comptime b = IntTuple(12)
    comptime c = IntTuple(8)
    assert_true(weakly_compatible(a, a))
    assert_true(weakly_compatible(b, b))
    assert_true(weakly_compatible(c, c))
    assert_false(weakly_compatible(a, b))
    assert_false(weakly_compatible(a, c))
    assert_true(weakly_compatible(c, a))

    comptime a0 = IntTuple(IntTuple(16))
    assert_true(weakly_compatible(a0, a0))
    assert_true(weakly_compatible(a, a0))
    assert_false(weakly_compatible(a0, a))
    assert_true(weakly_compatible(c, a0))
    assert_false(weakly_compatible(a0, c))
    assert_false(weakly_compatible(b, a0))
    assert_false(weakly_compatible(a0, b))

    comptime a1 = IntTuple(2, 8)
    assert_true(weakly_compatible(a1, a1))
    assert_true(weakly_compatible(a, a1))
    assert_false(weakly_compatible(a0, a1))
    assert_false(weakly_compatible(a1, a0))
    assert_true(weakly_compatible(a1, IntTuple(2, IntTuple(2, 4))))

    comptime a2 = IntTuple(IntTuple(2, 8))
    assert_true(weakly_compatible(a2, a2))
    assert_true(weakly_compatible(a, a2))
    assert_true(weakly_compatible(c, a2))
    assert_true(weakly_compatible(a0, a2))
    assert_false(weakly_compatible(a2, a0))

    comptime a3 = IntTuple(IntTuple(2, IntTuple(4, 2)))
    assert_true(weakly_compatible(a3, a3))
    assert_true(weakly_compatible(a, a3))
    assert_true(weakly_compatible(c, a3))
    assert_true(weakly_compatible(a0, a3))
    assert_false(weakly_compatible(a3, a0))
    assert_true(weakly_compatible(a2, a3))
    assert_false(weakly_compatible(a3, a2))


def test_fill_like() raises:
    print("== test_fill_like")
    comptime t1 = IntTuple(2, IntTuple(2, 2), IntTuple(1))
    comptime t2 = IntTuple(IntTuple(3, 4), 2, IntTuple(3))
    assert_equal(fill_like(t1, 0), IntTuple(0, IntTuple(0, 0), IntTuple(0)))
    assert_equal(fill_like(t2, 1), IntTuple(IntTuple(1, 1), 1, IntTuple(1)))


def test_reverse() raises:
    print("== test_reverse")
    comptime t1 = IntTuple(2, IntTuple(3, 4))
    comptime t2 = IntTuple(IntTuple(1, 2), 3, 4, IntTuple(5, 6, 7))
    assert_equal(reverse(t1), IntTuple(IntTuple(4, 3), 2))
    assert_equal(reverse(t2), IntTuple(IntTuple(7, 6, 5), 4, 3, IntTuple(2, 1)))


def test_depth() raises:
    print("== test_depth")
    assert_equal(depth(IntTuple(1)), 0)
    assert_equal(depth(IntTuple(1, 2)), 1)
    assert_equal(depth(IntTuple(1, IntTuple(2, 3))), 2)


def test_unknown_value_arith() raises:
    print("== test_unknown_value_arith")
    var t = IntTuple(UNKNOWN_VALUE, IntTuple(2, 3), 4)
    assert_equal(
        prefix_product(t),
        IntTuple(1, IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE), UNKNOWN_VALUE),
    )
    assert_equal(sum(t), UNKNOWN_VALUE)


def test_compact_order() raises:
    print("== test_compact_order")
    assert_equal(
        compact_order(IntTuple(2, 3, 4, 5), IntTuple(1, 4, 3, 5)),
        IntTuple(1, 8, 2, 24),
    )
    assert_equal(
        compact_order(
            IntTuple(2, IntTuple(3, 4), 5), IntTuple(1, IntTuple(2, 3), 4)
        ),
        IntTuple(1, IntTuple(2, 6), 24),
    )
    assert_equal(
        compact_order(IntTuple(2, 2, 2, 2), IntTuple(0, 2, 3, 1)),
        IntTuple(1, 4, 8, 2),
    )
    assert_equal(
        compact_order(IntTuple(2, 3, 4, 5), IntTuple(0, 2, 3, 1)),
        IntTuple(1, 10, 30, 2),
    )


def test_iter() raises:
    var a = IntTuple(2, 3)
    var it = iter(a)

    assert_equal(it.bounds()[0], 2)
    assert_equal(it.bounds()[1].value(), 2)

    assert_equal(Int(next(it)), 2)

    assert_equal(it.bounds()[0], 1)
    assert_equal(it.bounds()[1].value(), 1)

    assert_equal(Int(next(it)), 3)

    assert_equal(it.bounds()[0], 0)
    assert_equal(it.bounds()[1].value(), 0)

    with assert_raises():
        _ = next(it)  # raises StopIteration
    var b = IntTuple(4, 5, 6)
    var it2 = zip(a, b)
    var elem = next(it2)
    assert_equal(Int(elem[0]), 2)
    assert_equal(Int(elem[1]), 4)
    elem = next(it2)
    assert_equal(Int(elem[0]), 3)
    assert_equal(Int(elem[1]), 5)
    # zipping shortest
    with assert_raises():
        _ = next(it)  # raises StopIteration
    var c = IntTuple()
    with assert_raises():
        var it = iter(c)
        _ = it.__next__()  # raises StopIteration


def test_value_nested_tuple() raises:
    """Test that value() correctly extracts values from nested single-element tuples.
    """
    # Create a tuple where elements are stored as nested single-element tuples
    # This simulates what happens with Layout.row_major(8, 8).stride
    var stride = reverse(prefix_product(reverse(IntTuple(8, 8))))

    # Verify the tuple is constructed correctly
    assert_equal(String(stride), "(8, 1)")

    # Test that elements are stored as tuples, not direct values
    assert_true(stride.is_tuple(0))
    assert_true(stride.is_tuple(1))
    assert_false(stride.is_value(0))
    assert_false(stride.is_value(1))

    # Test that value() correctly extracts the actual integer values
    # Before the fix, this would return -65536 and -65537 (negative offsets)
    assert_equal(stride.value(0), 8)
    assert_equal(stride.value(1), 1)

    # Test with different sizes to ensure robustness
    var stride2 = reverse(prefix_product(reverse(IntTuple(4, 16))))
    assert_equal(stride2.value(0), 16)
    assert_equal(stride2.value(1), 1)

    var stride3 = reverse(prefix_product(reverse(IntTuple(3, 4, 5))))
    assert_equal(stride3.value(0), 20)
    assert_equal(stride3.value(1), 5)
    assert_equal(stride3.value(2), 1)

    # Test that __getitem__ also works correctly
    assert_equal(Int(stride[0]), 8)
    assert_equal(Int(stride[1]), 1)


def test_coord_to_int_tuple_conversion() raises:
    """`Coord` round-trips to `IntTuple` preserving nested structure."""
    var t = Coord(Coord(Idx[2], Idx[3]), Idx[4])
    var t2 = coord_to_int_tuple(t)
    assert_equal(t2[0][0], 2)
    assert_equal(t2[0][1], 3)
    assert_equal(t2[1], 4)


def main() raises:
    test_tuple_basic()
    test_tuple_slicing()
    test_tuple_basic_ops()
    test_sorted()

    test_apply()
    test_product()
    test_inner_product()
    test_shape_div()
    test_prefix_product()

    test_crd2idx()
    test_idx2crd()

    test_weakly_congruent()
    test_weakly_compatible()
    test_fill_like()
    test_reverse()
    test_depth()
    test_compact_order()

    test_unknown_value_arith()

    test_iter()
    test_value_nested_tuple()

    test_coord_to_int_tuple_conversion()
