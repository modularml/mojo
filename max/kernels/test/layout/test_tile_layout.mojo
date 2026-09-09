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

from layout import Idx, Coord, coord, row_major, col_major
from layout.tile_layout import (
    Layout,
    CoalesceLayout,
    blocked_product,
    coalesce,
    _format_layout,
    _print_layout,
    zipped_divide,
)
from std.testing import assert_equal, TestSuite


# ===----------------------------------------------------------------------=== #
# coalesce tests
# ===----------------------------------------------------------------------=== #


def test_coalesce_row_major_2d() raises:
    """Row-major (2, 4) strides (4, 1) does not coalesce.

    Pairs: (2, 4), (4, 1). Check: 2*4=8 != 1. No merge.
    Coalesce processes dimensions left to right; for row-major the
    stride-1 dim is last, so consecutive pairs are not contiguous.
    """
    var layout = row_major[2, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 4)
    assert_equal(c.shape[1]().value(), 4)
    assert_equal(c.stride[1]().value(), 1)


def test_coalesce_row_major_3d() raises:
    """Row-major (2, 3, 4) strides (12, 4, 1) does not coalesce.

    Pairs: (2, 12), (3, 4), (4, 1). No consecutive pair is contiguous.
    """
    var layout = row_major[2, 3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 12)
    assert_equal(c.shape[1]().value(), 3)
    assert_equal(c.stride[1]().value(), 4)
    assert_equal(c.shape[2]().value(), 4)
    assert_equal(c.stride[2]().value(), 1)


def test_coalesce_col_major_2d() raises:
    """Col-major (3, 4) strides (1, 3) -> coalesced to (12,) stride (1,).

    Flat pairs: (3, 1), (4, 3). Check: 3*1 == 3? Yes -> merge to (12, 1).
    """
    var layout = col_major[3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 12)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_col_major_3d() raises:
    """Col-major (2, 3, 4) strides (1, 2, 6) -> coalesced to (24,) stride (1,).

    Flat pairs: (2, 1), (3, 2), (4, 6).
    (2, 1): prev
    (3, 2): 2*1=2 == 2 -> merge: (6, 1)
    (4, 6): 6*1=6 == 6 -> merge: (24, 1)
    """
    var layout = col_major[2, 3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 24)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_already_1d() raises:
    """A 1D layout stays 1D after coalescing."""
    var layout = row_major[8]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 8)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_non_contiguous() raises:
    """Non-contiguous layout remains unchanged.

    Shape (2, 2), stride (4, 1) -> 2*4=8 != 1, so no merge.
    """
    var layout = Layout(
        shape=(Idx[2], Idx[2]),
        stride=(Idx[4], Idx[1]),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 4)
    assert_equal(c.shape[1]().value(), 2)
    assert_equal(c.stride[1]().value(), 1)


def test_coalesce_with_shape_1_dims() raises:
    """Shape-1 dimensions are removed.

    Shape (1, 4, 1), stride (8, 1, 2) -> skip shape-1 dims -> (4, 1).
    """
    var layout = Layout(
        shape=(Idx[1], Idx[4], Idx[1]),
        stride=(Idx[8], Idx[1], Idx[2]),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 4)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_nested_blocked_product() raises:
    """Coalesce a nested layout from blocked_product.

    block = row_major[4]() (shape (4,), stride (1,))
    tiler = row_major[3]() (shape (3,), stride (1,))
    blocked = blocked_product(block, tiler)
    shape: ((4,), (3,)), stride: ((1,), (4,))
    Flat: (4, 1), (3, 4). 4*1 == 4 -> merge to (12, 1).
    """
    var block = row_major[4]()
    var tiler = row_major[3]()
    var blocked = blocked_product(block, tiler)
    var c = coalesce(blocked)
    assert_equal(c.shape[0]().value(), 12)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_type_level() raises:
    """CoalesceLayout can be used at the type level."""
    comptime CM = type_of(col_major[3, 4]())
    comptime C = CoalesceLayout[CM]

    comptime assert C.static_shape[0] == 12
    comptime assert C.static_stride[0] == 1

    var layout = C()
    assert_equal(layout.shape[0]().value(), 12)
    assert_equal(layout.stride[0]().value(), 1)


def test_coalesce_partial_merge() raises:
    """Coalesce merges only contiguous dimensions.

    Shape (2, 4, 3), stride (16, 1, 4).
    Flat: (2, 16), (4, 1), (3, 4).
    (2, 16): prev
    (4, 1): 2*16=32 != 1, new dim -> (2, 16), (4, 1)
    (3, 4): 4*1=4 == 4, merge -> (2, 16), (12, 1)
    """
    var layout = Layout(
        shape=(Idx[2], Idx[4], Idx[3]),
        stride=(Idx[16], Idx[1], Idx[4]),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 16)
    assert_equal(c.shape[1]().value(), 12)
    assert_equal(c.stride[1]().value(), 1)


# ===----------------------------------------------------------------------=== #
# blocked_product with coalesce_output tests
# ===----------------------------------------------------------------------=== #


def test_blocked_product_coalesce_output_1d() raises:
    var block = row_major[4]()
    var tiler = row_major[3]()
    var result = blocked_product[coalesce_output=True](block, tiler)
    assert_equal(result.shape[0]().value(), 12)
    assert_equal(result.stride[0]().value(), 1)


def test_blocked_product_coalesce_output_no_coalesce() raises:
    """Test blocked_product with coalesce_output=True when no modes coalesce.

    Same as the existing test_coalesced_blocked_product_no_coalesce but
    exercising the function overload instead of the type alias.
    """
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var result = blocked_product[coalesce_output=True](block, tiler)

    # Mode 0: nested shape (2, 2), stride (2, 12)
    assert_equal(result.shape[0]()[0].value(), 2)
    assert_equal(result.shape[0]()[1].value(), 2)
    assert_equal(result.stride[0]()[0].value(), 2)
    assert_equal(result.stride[0]()[1].value(), 12)

    # Mode 1: nested shape (2, 3), stride (1, 4)
    assert_equal(result.shape[1]()[0].value(), 2)
    assert_equal(result.shape[1]()[1].value(), 3)
    assert_equal(result.stride[1]()[0].value(), 1)
    assert_equal(result.stride[1]()[1].value(), 4)


def test_blocked_product_coalesce_output_false() raises:
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var result = blocked_product[coalesce_output=False](block, tiler)

    # Should be nested, same as blocked_product(block, tiler)
    assert_equal(result.shape[0]()[0].value(), 2)
    assert_equal(result.shape[0]()[1].value(), 2)
    assert_equal(result.shape[1]()[0].value(), 2)
    assert_equal(result.shape[1]()[1].value(), 3)
    assert_equal(result.stride[0]()[0].value(), 2)
    assert_equal(result.stride[0]()[1].value(), 12)
    assert_equal(result.stride[1]()[0].value(), 1)
    assert_equal(result.stride[1]()[1].value(), 4)


# ===----------------------------------------------------------------------=== #
# idx2crd tests
# ===----------------------------------------------------------------------=== #


def test_idx2crd_flat_row_major() raises:
    """Test idx2crd on a flat row-major layout."""
    var layout = row_major[3, 4]()
    var c0 = layout.idx2crd(0)
    assert_equal(Int(c0[0].value()), 0)
    assert_equal(Int(c0[1].value()), 0)
    var c5 = layout.idx2crd(5)
    assert_equal(Int(c5[0].value()), 1)
    assert_equal(Int(c5[1].value()), 1)
    var c11 = layout.idx2crd(11)
    assert_equal(Int(c11[0].value()), 2)
    assert_equal(Int(c11[1].value()), 3)


def test_idx2crd_nested_zipped_divide() raises:
    """Test idx2crd on a nested layout from zipped_divide."""
    var base = row_major[6, 8]()
    var layout = zipped_divide[coord[2, 2]](base)

    # coord[1,1] maps to linear index 24
    var linear_idx = Int(layout(coord[1, 1]))
    assert_equal(linear_idx, 24)

    # Inverse: idx2crd(24) should give nested coords ((1, 0), (1, 0))
    var coords = layout.idx2crd(linear_idx)
    var inner = coords[0].tuple()
    var outer = coords[1].tuple()
    assert_equal(Int(inner[0].value()), 1)
    assert_equal(Int(inner[1].value()), 0)
    assert_equal(Int(outer[0].value()), 1)
    assert_equal(Int(outer[1].value()), 0)


def test_idx2crd_nested_roundtrip() raises:
    """Verify that crd2idx(idx2crd(i)) == i for all valid indices."""
    var base = row_major[4, 6]()
    var layout = zipped_divide[coord[2, 3]](base)

    for i in range(24):
        var coords = layout.idx2crd(i)
        var inner = coords[0].tuple()
        var outer = coords[1].tuple()
        # Reconstruct using the flat formula
        var reconstructed = (
            Int(inner[0].value()) * 6
            + Int(inner[1].value()) * 1
            + Int(outer[0].value()) * 12
            + Int(outer[1].value()) * 3
        )
        assert_equal(reconstructed, i)


# ===----------------------------------------------------------------------=== #
# print_layout / format_layout tests
# ===----------------------------------------------------------------------=== #


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
    _format_layout(
        Layout(shape=(Idx[4], Idx[4]), stride=(Idx[1], Idx[2])),
        output,
    )
    assert_equal(output, expected)


def test_format_layout_blocked() raises:
    var expected = """\
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  1 |  4 |  5 |
    +----+----+----+----+
 1  |  2 |  3 |  6 |  7 |
    +----+----+----+----+
 2  |  8 |  9 | 12 | 13 |
    +----+----+----+----+
 3  | 10 | 11 | 14 | 15 |
    +----+----+----+----+
"""

    var output = String()
    _format_layout(
        blocked_product(row_major[2, 2](), row_major[2, 2]()),
        output,
    )
    assert_equal(output, expected)


def test_print_layout() raises:
    # Flat 2×2 layout: shape (2,2), stride (1,2)
    _print_layout(Layout(shape=(Idx[2], Idx[2]), stride=(Idx[1], Idx[2])))

    # Nested 4×4 layout via blocked_product
    var l1 = blocked_product(row_major[2, 2](), row_major[2, 2]())
    _print_layout(l1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
