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

from std.collections import Array
from layout.layout_tensor import LayoutTensorIter
from layout import *
from layout._fillers import arange
from layout.layout import (
    MakeLayoutList,
    blocked_product,
    coalesce,
    complement,
    expand_modes_alike,
    format_layout,
    is_row_major,
    logical_divide,
    logical_product,
    right_inverse,
    size,
    sublayout,
    tile_to_shape,
    upcast,
    zipped_divide,
)
from std.math import ceildiv
from std.testing import assert_equal, assert_raises

from std.utils import IndexList


# CHECK-LABEL: test_layout_basic
def test_layout_basic() raises:
    print("== test_layout_basic")

    # Basic constructor
    comptime shape = IntTuple(2, IntTuple(3, IntTuple(4)))
    comptime stride = IntTuple(1, IntTuple(2, IntTuple(6)))
    var layout = Layout(shape, stride)
    assert_equal(
        layout, Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(1, IntTuple(2, 6)))
    )
    assert_equal(
        layout.make_shape_unknown[axis=0](),
        Layout(
            IntTuple(UNKNOWN_VALUE, IntTuple(3, 4)), IntTuple(1, IntTuple(2, 6))
        ),
    )
    assert_equal(
        layout.make_shape_unknown[axis=1](),
        Layout(
            IntTuple(2, IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)),
            IntTuple(1, IntTuple(2, 6)),
        ),
    )

    # Row major variadic input
    assert_equal(Layout.row_major(2, 3), Layout(IntTuple(2, 3), IntTuple(3, 1)))
    # Row major tuple input
    assert_equal(
        Layout.row_major(IntTuple(2, 3)), Layout(IntTuple(2, 3), IntTuple(3, 1))
    )
    assert_equal(
        Layout.row_major(IntTuple(2, IntTuple(3, 4))),
        Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(12, IntTuple(4, 1))),
    )

    assert_equal(Layout.col_major(2, 3), Layout(IntTuple(2, 3), IntTuple(1, 2)))

    # Check if layout is row_major
    assert_equal(is_row_major[3](Layout.row_major(3, 2, 3)), True)
    assert_equal(is_row_major[2](Layout.col_major(3, 3)), False)

    # test dynamic layout
    assert_equal(
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
        Layout.row_major[3](),
    )
    assert_equal(
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        Layout.row_major[2](),
    )

    # test index list construction
    assert_equal(
        Layout.row_major(IndexList[2](2, 3)),
        Layout.row_major(2, 3),
    )

    # testing col major
    assert_equal(
        Layout.col_major[3](IndexList[3](3, 64, 128)),
        Layout(IntTuple(3, 64, 128), IntTuple(1, 3, 192)),
    )

    assert_equal(
        Layout.col_major[3](IndexList[3](UNKNOWN_VALUE, 64, 128)),
        Layout(
            IntTuple(UNKNOWN_VALUE, 64, 128),
            IntTuple(1, UNKNOWN_VALUE, UNKNOWN_VALUE),
        ),
    )

    var idx_list = IndexList[3](32, 8, 16)
    assert_equal(
        Layout.col_major[3](idx_list),
        Layout(IntTuple(32, 8, 16), IntTuple(1, 32, 256)),
    )

    assert_equal(
        Layout.col_major[3](IndexList[3](UNKNOWN_VALUE, 8, 16)),
        Layout(
            IntTuple(UNKNOWN_VALUE, 8, 16),
            IntTuple(1, UNKNOWN_VALUE, UNKNOWN_VALUE),
        ),
    )

    assert_equal(
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, 128, 576),
        Layout(
            IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 128, 576),
            IntTuple(-1, 73728, 576, 1),
        ),
    )


def test_layout_stride_value_access() raises:
    """Test that Layout stride values can be accessed correctly via `value()` method.
    """
    # Test basic 2D row-major layout
    var layout_2d = Layout.row_major(8, 8)
    assert_equal(layout_2d.stride.value(0), 8)
    assert_equal(layout_2d.stride.value(1), 1)

    # Test different sizes
    var layout_4x16 = Layout.row_major(4, 16)
    assert_equal(layout_4x16.stride.value(0), 16)
    assert_equal(layout_4x16.stride.value(1), 1)

    # Test 3D row-major layout
    var layout_3d = Layout.row_major(3, 4, 5)
    assert_equal(layout_3d.stride.value(0), 20)
    assert_equal(layout_3d.stride.value(1), 5)
    assert_equal(layout_3d.stride.value(2), 1)

    # Test column-major layout (should also work)
    var col_layout = Layout.col_major(8, 8)
    assert_equal(col_layout.stride.value(0), 1)
    assert_equal(col_layout.stride.value(1), 8)

    # Test that shape values are also accessible
    assert_equal(layout_2d.shape.value(0), 8)
    assert_equal(layout_2d.shape.value(1), 8)


def test_unknowns() raises:
    print("== test_unknowns")
    comptime shape = IntTuple(2, IntTuple(UNKNOWN_VALUE, 4))
    comptime stride = IntTuple(1, IntTuple(2, 6))
    comptime layout = Layout(shape, stride)
    assert_equal(comptime (layout.shape.all_known()), False)
    assert_equal(comptime (layout.stride.all_known()), True)
    assert_equal(comptime (layout.all_dims_known()), False)


def validate_coalesce[layout: Layout]() raises:
    comptime layoutR = coalesce(layout)

    # print(layout, "=> ", layoutR)

    assert_equal(comptime (size(layoutR)), comptime (size(layout)))

    for i in range(comptime (size(layout))):
        assert_equal(materialize[layoutR]()(i), materialize[layout]()(i))


# CHECK-LABEL: test_coalesce
def test_coalesce() raises:
    print("== test_coalesce")

    validate_coalesce[
        Layout(IntTuple(2, IntTuple(1, 6)), IntTuple(1, IntTuple(6, 2)))
    ]()

    validate_coalesce[Layout(1, 0)]()

    validate_coalesce[Layout(1, 0)]()

    validate_coalesce[Layout(IntTuple(2, 4))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6), IntTuple(1, 6, 2))]()

    validate_coalesce[Layout(IntTuple(2, 1, 6), IntTuple(1, 7, 2))]()

    validate_coalesce[Layout(IntTuple(2, 1, 6), IntTuple(4, 7, 8))]()

    validate_coalesce[Layout(IntTuple(2, IntTuple(4, 6)))]()

    validate_coalesce[Layout(IntTuple(2, 4), IntTuple(4, 1))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6), IntTuple(24, 6, 1))]()

    validate_coalesce[Layout(IntTuple(2, 1, 3), IntTuple(2, 4, 4))]()

    validate_coalesce[
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(1, 4), IntTuple(8, 32)),
        )
    ]()

    # Validate keeping rank
    # CHECK: (16:4)
    print(coalesce(Layout(IntTuple(2, 8), IntTuple(4, 8))))
    # CHECK: ((2, 8):(4, 8))
    print(coalesce(Layout(IntTuple(2, 8), IntTuple(4, 8)), keep_rank=True))


def validate_composition[layoutA: Layout, layoutB: Layout]() raises:
    var layoutR = composition(materialize[layoutA](), materialize[layoutB]())

    # print(layoutA, "o", layoutB, "=>", layoutR)

    # True post-condition: Every coordinate c of layoutB with L1D(c) < size(layoutR) is a coordinate of layoutR.

    # Test that R(c) = A(B(c)) for all coordinates c in layoutR
    for i in range(size(layoutR)):
        assert_equal(
            layoutR(i),
            materialize[layoutA]()(materialize[layoutB]()(i)),
        )


# CHECK-LABEL: test_composition
def test_composition() raises:
    print("== test_composition")

    validate_composition[Layout(1, 0), Layout(1, 0)]()

    validate_composition[Layout(1, 0), Layout(1, 1)]()

    validate_composition[Layout(1, 1), Layout(1, 0)]()

    validate_composition[Layout(1, 1), Layout(1, 1)]()

    validate_composition[Layout(IntTuple(4)), Layout(IntTuple(4))]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(4), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(0)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(4), IntTuple(0))
    ]()

    validate_composition[
        Layout(IntTuple(1), IntTuple(0)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(1), IntTuple(0))
    ]()

    validate_composition[Layout(IntTuple(4)), Layout(IntTuple(2))]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(2), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(2), IntTuple(2))
    ]()

    validate_composition[Layout(IntTuple(12)), Layout(IntTuple(4, 3))]()

    validate_composition[
        Layout(IntTuple(12), IntTuple(2)), Layout(IntTuple(4, 3))
    ]()

    validate_composition[
        Layout(IntTuple(12)), Layout(IntTuple(4, 3), IntTuple(3, 1))
    ]()

    validate_composition[
        Layout(IntTuple(12), IntTuple(2)),
        Layout(IntTuple(4, 3), IntTuple(3, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(12)), Layout(IntTuple(2, 3), IntTuple(2, 4))
    ]()

    validate_composition[Layout(IntTuple(4, 3)), Layout(IntTuple(4, 3))]()

    validate_composition[Layout(IntTuple(4, 3)), Layout(IntTuple(12))]()

    validate_composition[
        Layout(IntTuple(4, 3)), Layout(IntTuple(6), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3)), Layout(IntTuple(6, 2), IntTuple(2, 1))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(4, 3))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(12))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(6), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)),
        Layout(IntTuple(6, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(8, 8)),
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
    ]()

    validate_composition[
        Layout(IntTuple(8, 8), IntTuple(8, 1)),
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
    ]()

    validate_composition[
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
        Layout(8, 4),
    ]()

    validate_composition[
        Layout(IntTuple(IntTuple(4, 2)), IntTuple(IntTuple(1, 16))),
        Layout(IntTuple(4, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(2, 2), IntTuple(2, 1)),
        Layout(IntTuple(2, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2)), Layout(IntTuple(2, 2, 2), IntTuple(2, 8, 1))
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2), IntTuple(2, 8, 1)),
        Layout(IntTuple(2, 2, 2), IntTuple(1, 8, 2)),
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2), IntTuple(2, 8, 1)),
        Layout(IntTuple(4, 2, 2), IntTuple(2, 8, 1)),
    ]()


# CHECK-LABEL: test_by_mode_composition
def test_by_mode_composition() raises:
    print("== test_by_mode_composition")

    # The correctness here is built on top of default composition, which has
    # been tested extensively above. Keep simple tests only.

    var layout0 = Layout.row_major(8, 4)
    var tiler = MakeLayoutList(Layout(4, 1), Layout(2, 1))
    assert_equal(
        composition(layout0^, tiler),
        Layout(IntTuple(4, 2), IntTuple(4, 1)),
    )

    var layout1 = Layout.row_major(IntTuple(IntTuple(8, 6), 4, 2))
    assert_equal(
        composition(layout1^, tiler),
        Layout(IntTuple(4, 2, 2), IntTuple(48, 2, 1)),
    )


def validate_complement[layout: Layout]() raises:
    comptime layoutR = complement(layout)

    # print(layout, " => ", layoutR)

    # Post-condition: test disjointness of the codomains
    for a in range(comptime (size(layout))):
        for b in range(comptime (size(layoutR))):
            assert_equal(
                (materialize[layout]()(a) != materialize[layoutR]()(b))
                or (
                    materialize[layout]()(a) == 0
                    and materialize[layoutR]()(b) == 0
                ),
                True,
            )


# CHECK-LABEL: test_complement
def test_complement() raises:
    print("== test_complement")
    comptime c0 = complement(Layout(4, 1), 24)
    assert_equal(String(materialize[c0]()), "(6:4)")
    assert_equal(String(complement(Layout(6, 4), 24)), "(4:1)")
    assert_equal(
        String(complement(Layout(IntTuple(4, 6), IntTuple(1, 4)), 24)), "(1:0)"
    )
    assert_equal(String(complement(Layout(4, 2), 24)), "((2, 3):(1, 8))")
    assert_equal(
        String(complement(Layout(IntTuple(2, 4), IntTuple(1, 6)), 24)), "(3:2)"
    )
    assert_equal(
        String(complement(Layout(IntTuple(2, 2), IntTuple(1, 6)), 24)),
        "((3, 2):(2, 12))",
    )

    validate_complement[Layout(1, 0)]()

    validate_complement[Layout(1, 1)]()

    validate_complement[Layout(4, 0)]()

    validate_complement[Layout(IntTuple(2, 4), IntTuple(1, 2))]()

    validate_complement[Layout(IntTuple(2, 3), IntTuple(1, 2))]()

    validate_complement[Layout(IntTuple(2, 4), IntTuple(1, 4))]()

    validate_complement[Layout(IntTuple(2, 4, 8), IntTuple(8, 1, 64))]()

    validate_complement[
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(1, 4), IntTuple(8, 32)),
        )
    ]()

    validate_complement[
        Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(3, IntTuple(1, 6)))
    ]()

    validate_complement[Layout(IntTuple(4, 6), IntTuple(1, 6))]()

    validate_complement[Layout(IntTuple(4, 10), IntTuple(1, 10))]()

    # When size is UNKNOWN_VALUE, the remainder dimension should also be
    # UNKNOWN_VALUE so that downstream code correctly treats it as dynamic.
    comptime c_unknown = complement(Layout(4, 1), UNKNOWN_VALUE)
    assert_equal(String(materialize[c_unknown]()), "(-1:4)")


# CHECK-LABEL: test_logcial_divide
def test_logcial_divide() raises:
    print("== test_logcial_divide")
    var ld0 = logical_divide(
        Layout(IntTuple(4, 2, 3), IntTuple(2, 1, 8)), Layout(4, 2)
    )
    assert_equal(String(ld0), "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        String(
            logical_divide(
                Layout(
                    IntTuple(9, IntTuple(4, 8)), IntTuple(59, IntTuple(13, 1))
                ),
                MakeLayoutList(
                    Layout(3, 3), Layout(IntTuple(2, 4), IntTuple(1, 8))
                ),
            ),
        ),
        "(((3, 3), ((2, 4), (2, 2))):((177, 59), ((13, 2), (26, 1))))",
    )


# CHECK-LABEL: test_logical_product
def test_logical_product() raises:
    print("== test_logical_product")
    var lp0 = logical_product(
        Layout(IntTuple(2, 2), IntTuple(4, 1)), Layout(6, 1)
    )
    assert_equal(String(lp0), "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        String(
            logical_product(
                Layout(IntTuple(2, 5), IntTuple(5, 1)),
                MakeLayoutList(Layout(3, 5), Layout(4, 6)),
            )
        ),
        "(((2, 3), (5, 4)):((5, 10), (1, 30)))",
    )


# CHECK-LABEL: test_blocked_product
def test_blocked_product() raises:
    print("== test_blocked_product")
    var bp0 = blocked_product(
        Layout(IntTuple(2, 5), IntTuple(5, 1)),
        Layout(IntTuple(3, 4), IntTuple(1, 3)),
    )
    assert_equal(String(bp0), "(((2, 3), (5, 4)):((5, 10), (1, 30)))")
    var cm_M = 8
    var cm_K = 8
    var core_matrix = Layout.row_major(cm_M, cm_K)
    var t_M = 2
    var t_K = 3
    var bp1 = blocked_product(core_matrix^, Layout.col_major(t_M, t_K))
    # ((cm_M,         t_M), (cm_K,               t_K)):
    # ((cm_K, cm_M * cm_K), (1,    t_M * cm_M * cm_K))
    var reference_bp1 = Layout(
        IntTuple(IntTuple(cm_M, t_M), IntTuple(cm_K, t_K)),
        IntTuple(IntTuple(cm_K, cm_M * cm_K), IntTuple(1, t_M * cm_M * cm_K)),
    )
    assert_equal(bp1, reference_bp1)

    comptime bp2 = blocked_product(
        Layout.row_major(128, 8), Layout.row_major(1, 4)
    )
    assert_equal(
        String(materialize[bp2]()), "(((128, 1), (8, 4)):((8, 0), (1, 1024)))"
    )

    comptime bp3 = blocked_product(
        Layout.row_major(128, 8),
        Layout.row_major(1, 4),
        coalesce_output=True,
    )
    assert_equal(String(materialize[bp3]()), "((128, (8, 4)):(8, (1, 1024)))")


def test_tile_to_shape() raises:
    print("== test_tile_to_shape")
    var a = Layout(IntTuple(2, 5), IntTuple(5, 1))
    var b = tile_to_shape(a.copy(), IntTuple(6, 20))
    assert_equal(String(b), "(((2, 3), (5, 4)):((5, 10), (1, 30)))")
    var b2 = tile_to_shape(a^, IntTuple(6, 20), IntTuple(1, 0))
    assert_equal(String(b2), "(((2, 3), (5, 4)):((5, 40), (1, 10)))")


# CHECK-LABEL: test_print_layout
# CHECK: ((2, 2):(1, 2))
# CHECK:       0   1
# CHECK:     +---+---+
# CHECK:  0  | 0 | 2 |
# CHECK:     +---+---+
# CHECK:  1  | 1 | 3 |
# CHECK:     +---+---+
# CHECK: (((2, 2), (2, 2)):((2, 8), (1, 4)))
# CHECK:        0    1    2    3
# CHECK:     +----+----+----+----+
# CHECK:  0  |  0 |  1 |  4 |  5 |
# CHECK:     +----+----+----+----+
# CHECK:  1  |  2 |  3 |  6 |  7 |
# CHECK:     +----+----+----+----+
# CHECK:  2  |  8 |  9 | 12 | 13 |
# CHECK:     +----+----+----+----+
# CHECK:  3  | 10 | 11 | 14 | 15 |
# CHECK:     +----+----+----+----+
def test_print_layout():
    print("== test_print_layout")
    var l0 = Layout(IntTuple(2, 2), IntTuple(1, 2))
    var l1 = Layout(
        IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
        IntTuple(IntTuple(2, 8), IntTuple(1, 4)),
    )
    print_layout(l0)
    print_layout(l1)


def test_format_layout_grid() raises:
    var expected = """\
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  2 |  4 |  6 |
    +----+----+----+----+
 1  |  1 |  3 |  5 |  7 |
    +----+----+----+----+
 2  |  2 |  4 |  6 |  8 |
    +----+----+----+----+
 3  |  3 |  5 |  7 |  9 |
    +----+----+----+----+
"""

    var output = String()

    format_layout(
        Layout(IntTuple(4, 4), IntTuple(1, 2)),
        output,
    )

    assert_equal(output, expected)


# CHECK-LABEL: test_zipped_divide
def test_zipped_divide() raises:
    print("== test_zipped_divide")
    var layout_4x4_row_major = Layout.row_major(4, 4)
    assert_equal(
        String(zipped_divide(layout_4x4_row_major, Layout(2, 1))),
        "((2, (2, 4)):(4, (8, 1)))",
    )
    var zd0 = zipped_divide(
        layout_4x4_row_major,
        MakeLayoutList(Layout(2, 1), Layout(2, 1)),
    )
    assert_equal(String(zd0), "(((2, 2), (2, 2)):((4, 1), (8, 2)))")

    # Resemble the case for distributing a tile over warp group.
    var tile_layout = Layout(
        IntTuple(IntTuple(16, 64), 4), IntTuple(IntTuple(128, 2), 2048)
    )
    var thread_layout = Layout(
        IntTuple(IntTuple(8, 4), 4), IntTuple(IntTuple(4, 1), 32)
    )
    assert_equal(
        String(zipped_divide(tile_layout, thread_layout)),
        "((((8, 4), 4), ((2, 16), 1)):(((128, 2), 2048), ((1024, 8), 0)))",
    )

    # Swizzle for cuda core matmul.
    tile_layout = Layout(IntTuple(16, 8), IntTuple(32, 4))
    thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 8), IntTuple(IntTuple(1, 16), 2)
    )
    assert_equal(
        String(zipped_divide(tile_layout, thread_layout)),
        "((((2, 2), 8), (4, 1)):(((32, 64), 4), (128, 0)))",
    )


# CHECK-LABEL: test_sublayout
def test_sublayout() raises:
    print("== test_sublayout")
    var layout_2x3x4 = Layout(IntTuple(2, 3, 4), IntTuple(12, 4, 1))
    assert_equal(String(sublayout(layout_2x3x4, 0, 2)), "((2, 4):(12, 1))")
    var layout_2x3x4_rank_2 = Layout(
        IntTuple(IntTuple(2, 3), 2, 4), IntTuple(IntTuple(12, 4), 4, 1)
    )
    assert_equal(
        String(sublayout(layout_2x3x4_rank_2, 0, 1)),
        "(((2, 3), 2):((12, 4), 4))",
    )


# CHECK-LABEL: test_crd2idx
def test_crd2idx() raises:
    print("== test_crd2idx")
    var l_4x4_row_major = Layout.row_major(4, 4)
    var l_4x4_col_major = Layout.col_major(4, 4)
    # CHECK: 0 (0, 0) (0, 0)
    # CHECK: 1 (0, 1) (1, 0)
    # CHECK: 2 (0, 2) (2, 0)
    # CHECK: 3 (0, 3) (3, 0)
    # CHECK: 4 (1, 0) (0, 1)
    # CHECK: 5 (1, 1) (1, 1)
    # CHECK: 6 (1, 2) (2, 1)
    # CHECK: 7 (1, 3) (3, 1)
    # CHECK: 8 (2, 0) (0, 2)
    # CHECK: 9 (2, 1) (1, 2)
    # CHECK: 10 (2, 2) (2, 2)
    # CHECK: 11 (2, 3) (3, 2)
    # CHECK: 12 (3, 0) (0, 3)
    # CHECK: 13 (3, 1) (1, 3)
    # CHECK: 14 (3, 2) (2, 3)
    # CHECK: 15 (3, 3) (3, 3)
    for i in range(16):
        print(i, l_4x4_row_major.idx2crd(i), l_4x4_col_major.idx2crd(i))


# CHECK-LABEL: test_expand_modes_alike
def test_expand_modes_alike() raises:
    print("== test_expand_modes_alike")
    comptime layout_0 = Layout(
        IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
        IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
    )
    comptime layout_1 = Layout(
        IntTuple(30, IntTuple(2, 2)), IntTuple(2, IntTuple(60, 1))
    )
    var ema0 = materialize[expand_modes_alike(layout_0, layout_1)]()
    # CHECK: (((3, (5, 2)), (2, 2)):((1, (24, 12)), (3, 6)))
    print(ema0[0])
    # CHECK: (((3, (5, 2)), (2, 2)):((2, (6, 30)), (60, 1)))
    print(ema0[1])

    comptime layout_2 = Layout(
        IntTuple(IntTuple(3, IntTuple(IntTuple(IntTuple(7, 11), 5), 2)), 4),
        IntTuple(
            IntTuple(1, IntTuple(IntTuple(IntTuple(120, 840), 24), 12)), 3
        ),
    )
    comptime layout_3 = Layout(IntTuple(2310, IntTuple(2, 2)))
    var ema1 = materialize[expand_modes_alike(layout_2, layout_3)]()
    # CHECK: (((3, (((7, 11), 5), 2)), (2, 2)):((1, (((120, 840), 24), 12)), (3, 6)))
    print(ema1[0])
    # CHECK: (((3, (((7, 11), 5), 2)), (2, 2)):((1, (((3, 21), 231), 1155)), (2310, 4620)))
    print(ema1[1])

    var ema2 = materialize[
        expand_modes_alike(Layout(IntTuple(2, 2), IntTuple(2, 1)), Layout(4))
    ]()
    # CHECK: ((2, 2):(2, 1))
    print(ema2[0])
    # CHECK: ((2, 2):(1, 2))
    print(ema2[1])

    var ema3 = materialize[
        expand_modes_alike(Layout(IntTuple(3, 4), IntTuple(2, 6)), Layout(12))
    ]()
    # CHECK: ((3, 4):(2, 6))
    print(ema3[0])
    # CHECK: ((3, 4):(1, 3))
    print(ema3[1])


def test_upcast() raises:
    print("== test_upcast")
    comptime scatter = Layout(IntTuple(4, 3), IntTuple(2, 4))
    var up2 = materialize[upcast(scatter, 2)]()
    assert_equal(String(up2), "((4, 3):(1, 2))")
    var up4 = materialize[upcast(scatter, 4)]()
    var up22 = upcast(up2^, 2)
    assert_equal(up4, up22)
    assert_equal(String(up4), "((2, 3):(1, 1))")
    comptime scatter2 = Layout(IntTuple(8, 1024), IntTuple(1024, 1))
    var up16 = materialize[upcast(scatter2, 16)]()
    assert_equal(String(up16), "((8, 64):(64, 1))")


def validate_right_inverse[layout: Layout]() raises:
    var rinv_layout = materialize[right_inverse(layout)]()
    for i in range(comptime (layout.size())):
        assert_equal(i, materialize[layout]()(rinv_layout(i)))


def test_right_inverse() raises:
    validate_right_inverse[
        Layout(
            IntTuple(2, IntTuple(3, IntTuple(4))),
            IntTuple(1, IntTuple(2, IntTuple(6))),
        )
    ]()
    validate_right_inverse[Layout.row_major(8, 4)]()
    validate_right_inverse[Layout.col_major(8, 4)]()
    validate_right_inverse[
        Layout(
            IntTuple(IntTuple(3, IntTuple(IntTuple(IntTuple(7, 11), 5), 2)), 4),
            IntTuple(
                IntTuple(1, IntTuple(IntTuple(IntTuple(120, 840), 24), 12)), 3
            ),
        )
    ]()
    validate_right_inverse[
        Layout(
            IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
            IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
        )
    ]()


# CHECK-LABEL: test_transpose
def test_transpose() raises:
    print("== test_transpose")

    # Test 2D transpose - row-major to column-major
    var row_major = Layout.row_major(3, 4)
    var transposed = row_major.transpose()
    assert_equal(transposed, Layout.col_major(4, 3))

    # Test column-major to row-major
    var col_major = Layout.col_major(3, 4)
    var trans_col = col_major.transpose()
    assert_equal(trans_col, Layout.row_major(4, 3))

    # Test custom 2D strides
    comptime custom = Layout(IntTuple(3, 5), IntTuple(7, 2))
    comptime trans_custom = custom.transpose()
    assert_equal(trans_custom.shape, IntTuple(5, 3))
    assert_equal(trans_custom.stride, IntTuple(2, 7))

    # Test 4D layout
    var layout4d = Layout.row_major(2, 3, 4, 5)
    var trans4d = layout4d.transpose()
    assert_equal(trans4d.shape, IntTuple(5, 4, 3, 2))
    assert_equal(trans4d.stride, IntTuple(1, 5, 20, 60))

    # Test nested layout - only top level transposed
    var nested = Layout(
        IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(12, 4), 1)
    )
    var trans_nested = nested.transpose()
    assert_equal(trans_nested.shape, IntTuple(4, IntTuple(2, 3)))
    assert_equal(trans_nested.stride, IntTuple(1, IntTuple(12, 4)))

    # Test 3-level nested layout
    comptime deep_nested = Layout(
        IntTuple(IntTuple(2, 3), IntTuple(4, 5), 6),
        IntTuple(IntTuple(30, 10), IntTuple(6, 1), 120),
    )
    comptime trans_deep = deep_nested.transpose()
    assert_equal(trans_deep.shape, IntTuple(6, IntTuple(4, 5), IntTuple(2, 3)))
    assert_equal(
        trans_deep.stride, IntTuple(120, IntTuple(6, 1), IntTuple(30, 10))
    )

    # Test 1D layout (should be unchanged)
    var layout1d = Layout(IntTuple(10), IntTuple(1))
    var trans1d = layout1d.transpose()
    assert_equal(trans1d, layout1d)

    # Test layout with zero strides
    var zero_stride = Layout(IntTuple(3, 4), IntTuple(0, 1))
    var trans_zero = zero_stride.transpose()
    assert_equal(trans_zero.shape, IntTuple(4, 3))
    assert_equal(trans_zero.stride, IntTuple(1, 0))

    # Test layout with unknown values
    comptime unknown_shape = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    comptime trans_unknown = unknown_shape.transpose()
    assert_equal(trans_unknown.shape, IntTuple(4, UNKNOWN_VALUE))
    assert_equal(trans_unknown.stride, IntTuple(1, 4))

    # Test empty layout
    comptime empty = Layout()
    comptime trans_empty = empty.transpose()
    assert_equal(trans_empty.shape, IntTuple())
    assert_equal(trans_empty.stride, IntTuple())

    # Test double transpose (idempotence)
    assert_equal(transposed.transpose(), row_major)
    assert_equal(trans4d.transpose(), layout4d)
    assert_equal(trans_nested.transpose(), nested)

    # Test memory mapping preservation for 2D
    comptime for i in range(3):
        comptime for j in range(4):
            var original_idx = row_major(IntTuple(i, j))
            var transposed_idx = transposed(IntTuple(j, i))
            assert_equal(original_idx, transposed_idx)

    # Test size preservation
    assert_equal(row_major.size(), transposed.size())
    assert_equal(layout4d.size(), trans4d.size())
    assert_equal(nested.size(), trans_nested.size())

    # Test cosize preservation for compact layouts
    assert_equal(row_major.cosize(), transposed.cosize())
    assert_equal(col_major.cosize(), trans_col.cosize())


def test_iter() raises:
    var layout = Layout.row_major(1, 2, 3, 4)
    var it = iter(layout)
    assert_equal(next(it), Layout(1, 24))
    assert_equal(next(it), Layout(2, 12))
    assert_equal(next(it), Layout(3, 4))
    assert_equal(next(it), Layout(4, 1))
    with assert_raises():
        _ = next(it)  # raises StopIteration
    var layout2 = Layout()
    with assert_raises():
        var it = iter(layout2)
        _ = it.__next__()  # raises StopIteration


def test_arange_nested_layout() raises:
    """Test arange function with nested layout structures."""
    # Test nested layout with tile structure similar to GPU shared memory tiles
    var nested_tensor = LayoutTensor[
        .float32,
        Layout(
            IntTuple(IntTuple(16, 8), IntTuple(32, 2)),
            IntTuple(IntTuple(32, 1024), IntTuple(1, 512)),
        ),
        MutAnyOrigin,
        alignment=16,
    ].stack_allocation()
    arange(nested_tensor)

    # Test simple 2D layout with row-major for comparison
    var simple_tensor = LayoutTensor[
        .float32,
        Layout.row_major(4, 4),
        MutAnyOrigin,
    ].stack_allocation()
    arange(simple_tensor)

    # Verify values are filled in logical order (row-major)
    assert_equal(simple_tensor[0, 0], 0.0)
    assert_equal(simple_tensor[0, 1], 1.0)
    assert_equal(simple_tensor[0, 2], 2.0)
    assert_equal(simple_tensor[0, 3], 3.0)
    assert_equal(simple_tensor[1, 0], 4.0)
    assert_equal(simple_tensor[1, 1], 5.0)
    assert_equal(simple_tensor[1, 2], 6.0)
    assert_equal(simple_tensor[1, 3], 7.0)

    # Test column-major layout
    var col_major_tensor = LayoutTensor[
        .float32,
        Layout.col_major(4, 4),
        MutAnyOrigin,
    ].stack_allocation()
    arange(col_major_tensor)

    # For column-major, values should still be in logical row-major order
    # when accessed via [i, j] indexing
    assert_equal(col_major_tensor[0, 0], 0.0)
    assert_equal(col_major_tensor[0, 1], 1.0)
    assert_equal(col_major_tensor[1, 0], 4.0)
    assert_equal(col_major_tensor[1, 1], 5.0)


def test_layout_tensor_iterator_print() raises:
    """Test case for MSTDL-1984: Tensors generated from LayoutTensorIter won't print.
    """
    comptime buf_size = 16
    var storage = Array[Int16, buf_size](
        fill_with=lambda (i: Int) -> Int16: Int16(i)
    )
    comptime tile_layout = Layout.row_major(2, 2)
    var iter = LayoutTensorIter[
        DType.int16,
        tile_layout,
        masked=True,
    ](storage.unsafe_ptr(), buf_size)

    for _i in range(ceildiv(buf_size, comptime (tile_layout.size()))):
        var tile = iter[]
        # CHECK: 0 1
        # CHECK-NEXT: 2 3
        print(tile)
        # CHECK: runtime_layout.size(): 4
        print("  runtime_layout.size():", tile.runtime_layout.size())
        iter += 1
        # CHECK: 4 5
        # CHECK-NEXT: 6 7
        # CHECK: runtime_layout.size(): 4
        # CHECK: 8 9
        # CHECK-NEXT: 10 11
        # CHECK: runtime_layout.size(): 4
        # CHECK: 12 13
        # CHECK-NEXT: 14 15
        # CHECK: runtime_layout.size(): 4


def main() raises:
    test_layout_basic()
    test_layout_stride_value_access()
    test_unknowns()
    test_coalesce()
    test_composition()
    test_by_mode_composition()
    test_complement()
    test_logcial_divide()
    test_logical_product()
    test_blocked_product()
    test_tile_to_shape()
    test_print_layout()
    test_format_layout_grid()
    test_zipped_divide()
    test_sublayout()
    test_crd2idx()
    test_expand_modes_alike()
    test_upcast()
    test_right_inverse()
    test_transpose()
    test_iter()
    test_arange_nested_layout()
    test_layout_tensor_iterator_print()
