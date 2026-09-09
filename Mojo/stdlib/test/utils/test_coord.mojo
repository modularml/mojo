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
"""Tests for `std.utils.coord` (`Coord`, `idx2crd`, `crd2idx`, etc.)."""

from std.builtin.device_passable import DevicePassable
from std.sys import size_of
from std.testing import TestSuite, assert_equal, assert_true
from std.utils.coord import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    _Idx2CrdResultTypes,
    coord,
    crd2idx,
    idx2crd,
)


def test_nested_layouts() raises:
    # Create nested layouts
    var inner = Coord(Idx[2], Int(3))
    var nested = Coord(inner, Idx[4])
    assert_equal(inner[1].value(), 3)
    assert_equal(nested[0][0].value(), 2)
    assert_equal(nested[1].value(), 4)
    assert_equal(size_of[type_of(inner)](), size_of[Int]())
    assert_equal(size_of[type_of(nested)](), size_of[Int]())


def test_list_literal_construction() raises:
    var t = Coord[ComptimeInt[2], Int](
        Idx[2],
        Int(3),
    )
    assert_equal(t[0].value(), 2)
    assert_equal(t[1].value(), 3)


def test_flatten_empty() raises:
    var t = Coord[]()
    assert_true(t.flatten() == t)


def test_construction_from_int_variadic_empty() raises:
    var t = coord[]
    assert_equal(len(t), 0)


def test_construction_from_int_variadic() raises:
    var t = coord[1, 2, 3]
    assert_equal(len(t), 3)
    assert_equal(t[0].value(), 1)
    assert_equal(t[1].value(), 2)
    assert_equal(t[2].value(), 3)


def test_construction_from_int_variadic_list() raises:
    var t = Coord(Int32(1), Int32(2), Int32(3))
    assert_equal(len(t), 3)
    assert_equal(t[0].value(), 1)
    assert_equal(t[1].value(), 2)
    assert_equal(t[2].value(), 3)


def test_construction_from_array() raises:
    var a: Array[Int, 3] = [1, 2, 3]
    var t = Coord(a)
    assert_equal(len(t), 3)
    assert_equal(t[0].value(), 1)
    assert_equal(t[1].value(), 2)
    assert_equal(t[2].value(), 3)
    # An `Array` carries no compile-time extents, so every element of the
    # resulting `Coord` must be dynamic even when the values look constant.
    assert_true(not type_of(t).all_dims_known)
    assert_true(not type_of(t).ParamListType[0].is_static_value)


def test_construction_from_array_narrow_dtype() raises:
    var a: Array[Int32, 2] = [7, 9]
    var t = Coord(a)
    assert_true(type_of(t[0]) == Int32)
    assert_equal(t[0].value(), 7)
    assert_equal(t[1].value(), 9)


def test_static_product() raises:
    comptime p = coord[1, 2, 3].static_product
    assert_equal(p, 6)


def test_default_init() raises:
    var c = Coord[
        ComptimeInt[5],
        Int32,
        ComptimeInt[3],
        Int64,
    ]()
    assert_equal(c[0].value(), 5)
    assert_equal(c[1].value(), 0)
    assert_equal(c[2].value(), 3)
    assert_equal(c[3].value(), 0)


def test_default_init_nested() raises:
    var c = Coord[ComptimeInt[5], Coord[Int32, ComptimeInt[3]], Int64]()
    assert_equal(c[0].value(), 5)
    assert_equal(c[1][0].value(), 0)
    assert_equal(c[1][1].value(), 3)
    assert_equal(c[2].value(), 0)


def test_idx2crd_basic() raises:
    """Test basic idx2crd correctness with row-major layout."""
    var shape = Coord(Idx[3], Idx[4])
    var stride = Coord(Idx[4], Idx[1])

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 1)

    var c11 = idx2crd(11, shape, stride)
    assert_equal(c11[0].value(), 2)
    assert_equal(c11[1].value(), 3)


def test_idx2crd_static_shape_1() raises:
    """When a shape dim is statically 1, the coordinate is ComptimeInt[0]."""
    var shape = Coord(Idx[1], Idx[4])
    var stride = Coord(Idx[4], Idx[1])

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    var c3 = idx2crd(3, shape, stride)
    assert_equal(c3[0].value(), 0)
    assert_equal(c3[1].value(), 3)

    # First dim should be ComptimeInt[0] (static shape 1).
    assert_true(type_of(c0[0]) == ComptimeInt[0])
    # Second dim should be Scalar.
    assert_true(type_of(c0[1]) == Int64)


def test_idx2crd_all_static_1() raises:
    """When all shape dims are 1, all coordinates are ComptimeInt[0]."""
    var shape = Coord(Idx[1], Idx[1])
    var stride = Coord(Idx[1], Idx[1])

    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    assert_true(type_of(c0[0]) == ComptimeInt[0])
    assert_true(type_of(c0[1]) == ComptimeInt[0])


def test_idx2crd_mixed_static_dynamic() raises:
    """Shape (3, 1, 4): middle dim is statically 1, others are dynamic."""
    var shape = Coord(Idx[3], Idx[1], Idx[4])
    var stride = Coord(Idx[4], Idx[4], Idx[1])

    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 0)
    assert_equal(c5[2].value(), 1)

    assert_true(type_of(c5[0]) == Int64)
    assert_true(type_of(c5[1]) == ComptimeInt[0])
    assert_true(type_of(c5[2]) == Int64)


def test_idx2crd_no_static_1() raises:
    """When no shape dim is 1, all coordinates are Scalar."""
    var shape = Coord(Idx[3], Idx[4])
    var stride = Coord(Idx[4], Idx[1])

    var _c = idx2crd(0, shape, stride)
    assert_true(type_of(_c[0]) == Int64)
    assert_true(type_of(_c[1]) == Int64)


def test_idx2crd_result_types_runtime_idx() raises:
    """No shape-1 dims with runtime idx: all Scalar."""
    comptime shape = TypeList.of[ComptimeInt[3], ComptimeInt[4]]()
    comptime stride = TypeList.of[ComptimeInt[4], ComptimeInt[1]]()
    comptime types = _Idx2CrdResultTypes[.int64, Int64, stride, shape]
    assert_true(types[0] == Int64)
    assert_true(types[1] == Int64)


def test_idx2crd_result_types_shape_1() raises:
    """Shape dim of 1 produces ComptimeInt[0], others Scalar."""
    comptime shape = TypeList.of[ComptimeInt[1], ComptimeInt[4]]()
    comptime stride = TypeList.of[ComptimeInt[4], ComptimeInt[1]]()
    comptime types = _Idx2CrdResultTypes[.int64, Int64, stride, shape]
    assert_true(types[0] == ComptimeInt[0])
    assert_true(types[1] == Int64)


def test_idx2crd_result_types_all_shape_1() raises:
    """All shape dims are 1: all ComptimeInt[0]."""
    comptime shape = TypeList.of[ComptimeInt[1], ComptimeInt[1]]()
    comptime stride = TypeList.of[ComptimeInt[1], ComptimeInt[1]]()
    comptime types = _Idx2CrdResultTypes[.int64, Int64, stride, shape]
    assert_true(types[0] == ComptimeInt[0])
    assert_true(types[1] == ComptimeInt[0])


def test_idx2crd_result_types_runtime_shape() raises:
    """Scalar shape dims always produce Scalar result."""
    comptime shape = TypeList.of[Int, Int]()
    comptime stride = TypeList.of[Int, Int]()
    comptime types = _Idx2CrdResultTypes[.int64, Int64, stride, shape]
    assert_true(types[0] == Int64)
    assert_true(types[1] == Int64)


def test_idx2crd_result_types_all_static() raises:
    """All three static (idx=5, shape=(3,4), stride=(4,1)): compile-time results.
    """
    comptime shape = TypeList.of[ComptimeInt[3], ComptimeInt[4]]()
    comptime stride = TypeList.of[ComptimeInt[4], ComptimeInt[1]]()
    comptime types = _Idx2CrdResultTypes[
        DType.int64, ComptimeInt[5], stride, shape
    ]
    # (5 // 4) % 3 = 1
    assert_true(types[0] == ComptimeInt[1])
    # (5 // 1) % 4 = 1
    assert_true(types[1] == ComptimeInt[1])


def test_idx2crd_single_dim() raises:
    """Test idx2crd with a single (non-tuple) shape."""
    var c = idx2crd(7, Idx[10], Idx[1])
    assert_equal(c[0].value(), 7)
    assert_true(type_of(c[0]) == Int64)

    # Single dim with shape 1 should produce ComptimeInt[0].
    var c1 = idx2crd(0, Idx[1], Idx[1])
    assert_equal(c1[0].value(), 0)
    assert_true(type_of(c1[0]) == ComptimeInt[0])


def test_idx2crd_col_major() raises:
    """Test idx2crd with col-major strides (which was broken with sequential algorithm).
    """
    # Shape (3, 4), col-major strides (1, 3)
    var shape = Coord(Idx[3], Idx[4])
    var stride = Coord(Idx[1], Idx[3])

    # idx=0 -> (0, 0)
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)

    # idx=1 -> (1 // 1) % 3 = 1, (1 // 3) % 4 = 0 -> (1, 0)
    var c1 = idx2crd(1, shape, stride)
    assert_equal(c1[0].value(), 1)
    assert_equal(c1[1].value(), 0)

    # idx=3 -> (3 // 1) % 3 = 0, (3 // 3) % 4 = 1 -> (0, 1)
    var c3 = idx2crd(3, shape, stride)
    assert_equal(c3[0].value(), 0)
    assert_equal(c3[1].value(), 1)

    # idx=5 -> (5 // 1) % 3 = 2, (5 // 3) % 4 = 1 -> (2, 1)
    var c5 = idx2crd(5, shape, stride)
    assert_equal(c5[0].value(), 2)
    assert_equal(c5[1].value(), 1)

    # idx=11 -> (11 // 1) % 3 = 2, (11 // 3) % 4 = 3 -> (2, 3)
    var c11 = idx2crd(11, shape, stride)
    assert_equal(c11[0].value(), 2)
    assert_equal(c11[1].value(), 3)


def test_idx2crd_comptime_idx() raises:
    """Test idx2crd with a compile-time index producing compile-time results."""
    var shape = Coord(Idx[3], Idx[4])
    var stride = Coord(Idx[4], Idx[1])

    # Compile-time idx=5 with static shape and stride should yield ComptimeInt results.
    var c5 = idx2crd(Idx[5], shape, stride)
    # (5 // 4) % 3 = 1
    assert_equal(c5[0].value(), 1)
    # (5 // 1) % 4 = 1
    assert_equal(c5[1].value(), 1)
    # Both dimensions should be ComptimeInt.
    assert_true(type_of(c5[0]) == ComptimeInt[1])
    assert_true(type_of(c5[1]) == ComptimeInt[1])

    # Compile-time idx=0 should yield ComptimeInt[0] for both dims.
    var c0 = idx2crd(Idx[0], shape, stride)
    assert_equal(c0[0].value(), 0)
    assert_equal(c0[1].value(), 0)
    assert_true(type_of(c0[0]) == ComptimeInt[0])
    assert_true(type_of(c0[1]) == ComptimeInt[0])


def test_idx2crd_mixed_static_dynamic_idx() raises:
    """Test idx2crd with static idx but one runtime stride dimension."""
    # shape=(3, 4), stride=(Scalar, ComptimeInt[1])
    var shape = Coord(Idx[3], Idx[4])
    var stride = Coord[Int, ComptimeInt[1]](Int(4), Idx[1])

    # Static idx=5, but first stride is runtime -> first dim is Scalar.
    # Second stride is static, shape is static, idx is static -> ComptimeInt.
    var c5 = idx2crd(Idx[5], shape, stride)
    assert_equal(c5[0].value(), 1)
    assert_equal(c5[1].value(), 1)
    assert_true(type_of(c5[0]) == Int64)
    # (5 // 1) % 4 = 1
    assert_true(type_of(c5[1]) == ComptimeInt[1])


def test_idx2crd_nested_depth2() raises:
    """Test idx2crd with depth-2 nested shape: Coord(Coord(2, 3), Coord(4, 5)).

    Each sub-element gets the same idx, and the leaf formula is
    (idx // stride) % shape.
    """
    # Shape: ((2, 3), (4, 5)), Stride: ((60, 20), (5, 1))
    # This represents a 4D layout with strides grouped into two pairs.
    var shape = Coord(
        Coord(Idx[2], Idx[3]),
        Coord(Idx[4], Idx[5]),
    )
    var stride = Coord(
        Coord(Idx[60], Idx[20]),
        Coord(Idx[5], Idx[1]),
    )

    # idx=0 -> ((0,0), (0,0))
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0][0].value(), 0)
    assert_equal(c0[0][1].value(), 0)
    assert_equal(c0[1][0].value(), 0)
    assert_equal(c0[1][1].value(), 0)

    # idx=27 -> ((27//60)%2=0, (27//20)%3=1), ((27//5)%4=1, (27//1)%5=2))
    var c27 = idx2crd(27, shape, stride)
    assert_equal(c27[0][0].value(), 0)
    assert_equal(c27[0][1].value(), 1)
    assert_equal(c27[1][0].value(), 1)
    assert_equal(c27[1][1].value(), 2)

    # idx=65 -> ((65//60)%2=1, (65//20)%3=0), ((65//5)%4=1, (65//1)%5=0))
    var c65 = idx2crd(65, shape, stride)
    assert_equal(c65[0][0].value(), 1)
    assert_equal(c65[0][1].value(), 0)  # 65//20 = 3, 3%3 = 0
    assert_equal(c65[1][0].value(), 1)  # 65//5 = 13, 13%4 = 1
    assert_equal(c65[1][1].value(), 0)  # 65%5 = 0


def test_idx2crd_nested_depth3() raises:
    """Test idx2crd with depth-3 nested shape: Coord(Coord(Coord(2, 3), 4))."""
    var shape = Coord(
        Coord(Coord(Idx[2], Idx[3]), Idx[4]),
    )
    var stride = Coord(
        Coord(Coord(Idx[12], Idx[4]), Idx[1]),
    )

    # idx=0 -> (((0, 0), 0))
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0][0][0].value(), 0)
    assert_equal(c0[0][0][1].value(), 0)
    assert_equal(c0[0][1].value(), 0)

    # idx=7 -> (((7//12)%2=0, (7//4)%3=1), (7//1)%4=3))
    var c7 = idx2crd(7, shape, stride)
    assert_equal(c7[0][0][0].value(), 0)
    assert_equal(c7[0][0][1].value(), 1)
    assert_equal(c7[0][1].value(), 3)

    # idx=15 -> (((15//12)%2=1, (15//4)%3=0), (15//1)%4=3))
    var c15 = idx2crd(15, shape, stride)
    assert_equal(c15[0][0][0].value(), 1)
    assert_equal(c15[0][0][1].value(), 0)  # 15//4 = 3, 3%3 = 0
    assert_equal(c15[0][1].value(), 3)  # 15%4 = 3


def test_idx2crd_nested_depth4() raises:
    """Test idx2crd with depth-4 nested shape."""
    # Shape: ((2, 3), ((4, 5), 6)) — depth 4 at the deepest path
    # Using stride-per-element formula: (idx // stride) % shape
    var shape = Coord(
        Coord(Idx[2], Idx[3]),
        Coord(Coord(Idx[4], Idx[5]), Idx[6]),
    )
    var stride = Coord(
        Coord(Idx[360], Idx[120]),
        Coord(Coord(Idx[30], Idx[6]), Idx[1]),
    )

    # idx=0 -> ((0, 0), ((0, 0), 0))
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0][0].value(), 0)
    assert_equal(c0[0][1].value(), 0)
    assert_equal(c0[1][0][0].value(), 0)
    assert_equal(c0[1][0][1].value(), 0)
    assert_equal(c0[1][1].value(), 0)

    # idx=37:
    #   (37//360)%2=0, (37//120)%3=0
    #   (37//30)%4=1, (37//6)%5=1, (37//1)%6=1
    var c37 = idx2crd(37, shape, stride)
    assert_equal(c37[0][0].value(), 0)
    assert_equal(c37[0][1].value(), 0)
    assert_equal(c37[1][0][0].value(), 1)
    assert_equal(c37[1][0][1].value(), 1)  # 37//6=6, 6%5=1
    assert_equal(c37[1][1].value(), 1)  # 37%6=1


def test_idx2crd_nested_depth4_linear() raises:
    """Test idx2crd with a linear chain of 4 nested Coords."""
    var shape = Coord(
        Coord(Coord(Coord(Idx[2], Idx[3]), Idx[4]), Idx[5]),
    )
    var stride = Coord(
        Coord(Coord(Coord(Idx[60], Idx[20]), Idx[5]), Idx[1]),
    )

    # idx=0 -> ((((0, 0), 0), 0))
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0][0][0][0].value(), 0)
    assert_equal(c0[0][0][0][1].value(), 0)
    assert_equal(c0[0][0][1].value(), 0)
    assert_equal(c0[0][1].value(), 0)

    # idx=27 -> ((((27//60)%2=0, (27//20)%3=1), (27//5)%4=1), (27//1)%5=2))
    var c27 = idx2crd(27, shape, stride)
    assert_equal(c27[0][0][0][0].value(), 0)
    assert_equal(c27[0][0][0][1].value(), 1)
    assert_equal(c27[0][0][1].value(), 1)  # 27//5 = 5, 5%4 = 1
    assert_equal(c27[0][1].value(), 2)  # 27%5 = 2


def test_idx2crd_nested_mixed_flat_and_nested() raises:
    """Test idx2crd with a mix of flat and nested elements at depth 2."""
    # Shape: (Coord(2, 3), 4) — first element nested, second flat.
    var shape = Coord(
        Coord(Idx[2], Idx[3]),
        Idx[4],
    )
    var stride = Coord(
        Coord(Idx[12], Idx[4]),
        Idx[1],
    )

    # idx=0 -> ((0, 0), 0)
    var c0 = idx2crd(0, shape, stride)
    assert_equal(c0[0][0].value(), 0)
    assert_equal(c0[0][1].value(), 0)
    assert_equal(c0[1].value(), 0)

    # idx=7 -> (((7//12)%2=0, (7//4)%3=1), (7//1)%4=3)
    var c7 = idx2crd(7, shape, stride)
    assert_equal(c7[0][0].value(), 0)
    assert_equal(c7[0][1].value(), 1)
    assert_equal(c7[1].value(), 3)


def test_crd2idx_default_int64_over_32_bits() raises:
    """`crd2idx` with default `out_type=DType.int64` computes indices past 2^32.

    The dot product is evaluated at `out_type` precision (narrow-first multiply
    per `_crd2idx_flat`), so the default `int64` comfortably covers layouts
    whose linear index exceeds 32 bits — including individual `coord * stride`
    terms that themselves cross the 32-bit boundary. This is the common case
    for large tensors (e.g. 2GB+ row-major buffers) and must keep working.

    Narrow-`out_type` wraparound is not tested here. The old
    `Scalar[out_type](Int(a) * Int(b))` and the new
    `Scalar[out_type](a) * Scalar[out_type](b)` yield the same runtime
    value under standard two's complement narrowing (both equal
    `(a * b) mod 2^width`), so the behavioral change between the two
    implementations is not value-observable — the win is that the new
    form keeps the multiply at the narrow IR width, avoiding a hidden
    64-bit op in GPU codegen. That is an IR-shape property, not a value
    property, and is covered by inspection / ad-hoc codegen review rather
    than a unit test.
    """
    # Shape (8192, 8192) row-major. Max index = 8191 * 8192 + 8191 = 67_100_671,
    # still under 2^32, but we exercise a larger layout below.
    # Use shape (65536, 65536) with stride (65536, 1) so that max coord
    # i * 65536 crosses 2^32 at i >= 65536. Pick i = 131072 to force overflow.
    comptime OUTER = 200000
    comptime INNER = 65536
    var shape = Coord(Int(OUTER), Int(INNER))
    var stride = Coord(Int(INNER), Int(1))

    # Pick a coord whose single term `i * INNER` exceeds 2^32:
    # 131072 * 65536 = 2^33 = 8_589_934_592.
    comptime I = 131072
    comptime J = 12345
    var crd = Coord(Int(I), Int(J))

    comptime expected: Int = I * INNER + J * 1
    # Sanity: confirm this really does cross 2^32.
    assert_true(expected > (Int(1) << 32))

    var idx = crd2idx(crd, shape, stride)
    assert_equal(Int(idx), expected)


def test_crd2idx_narrow_uint32_within_range() raises:
    """`crd2idx[out_type=DType.uint32]` gives correct results when values fit.

    The narrow-first multiply design means a `uint32` `out_type` keeps the
    entire dot product at 32-bit precision — callers that pick it are
    responsible for keeping every `coord * stride` term and the sum within
    `uint32`. This test pins down that contract: for layouts that fit, the
    result matches the `int64` reference.
    """
    # 1024x1024 row-major: max index 1024*1023 + 1023 = 1_048_575 — well
    # under 2^32.
    comptime DIM = 1024
    var shape = Coord(Int(DIM), Int(DIM))
    var stride = Coord(Int(DIM), Int(1))

    comptime I = 777
    comptime J = 543
    var crd = Coord(Int(I), Int(J))

    comptime expected: Int = I * DIM + J

    var idx_u32 = crd2idx[out_type=DType.uint32](crd, shape, stride)
    assert_equal(Int(idx_u32), expected)

    var idx_i64 = crd2idx(crd, shape, stride)
    assert_equal(Int(idx_i64), expected)
    assert_equal(Int(idx_u32), Int(idx_i64))


def test_cast_preserves_static_dims() raises:
    var c = Coord[
        ComptimeInt[5],
        Int32,
        ComptimeInt[3],
        Int64,
    ](Idx[5], Int32(7), Idx[3], Int64(11))

    var casted = c.cast[.uint32]()
    assert_equal(casted[0].value(), 5)
    assert_equal(casted[1].value(), 7)
    assert_equal(casted[2].value(), 3)
    assert_equal(casted[3].value(), 11)
    assert_true(type_of(casted[0]) == ComptimeInt[5])
    assert_true(type_of(casted[1]) == UInt32)
    assert_true(type_of(casted[2]) == ComptimeInt[3])
    assert_true(type_of(casted[3]) == UInt32)


def test_device_passable_conformance() raises:
    # Encoder-driven bit-copy behavior is covered in `test_device_passable.mojo`.
    comptime assert conforms_to(Coord[ComptimeInt[2], Int], DevicePassable)
    comptime assert conforms_to(Coord[Int32, Int32], DevicePassable)
    comptime assert Coord[Int, Int].device_type == Coord[Int, Int]


def test_replace_keeps_other_dims_static() raises:
    var c = Coord(ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]())
    var moved = c.replace[1](Int64(7))

    assert_equal(Int(moved[0].value()), 3)
    assert_equal(Int(moved[1].value()), 7)
    assert_equal(Int(moved[2].value()), 5)

    # Only the replaced element gains runtime storage; `make_dynamic` would
    # have made all three dynamic.
    comptime assert size_of[type_of(moved)]() == size_of[Int64]()
    comptime assert (
        size_of[type_of(c.make_dynamic[DType.int64]())]()
        == 3 * size_of[Int64]()
    )


def test_replace_static_element_stays_comptime() raises:
    var c = Coord(ComptimeInt[8](), ComptimeInt[16](), ComptimeInt[128]())
    var idx = c.replace[2](Int64(64))

    # The untouched leading dims are still compile-time values.
    comptime assert type_of(idx).element_types[0] == ComptimeInt[8]
    comptime assert type_of(idx).element_types[1] == ComptimeInt[16]
    assert_equal(Int(idx[2].value()), 64)


def test_replace_first_and_last() raises:
    var c = Coord(ComptimeInt[2](), ComptimeInt[3](), ComptimeInt[4]())

    var first = c.replace[0](Int32(9))
    assert_equal(Int(first[0].value()), 9)
    assert_equal(Int(first[2].value()), 4)

    var last = c.replace[2](Int32(9))
    assert_equal(Int(last[0].value()), 2)
    assert_equal(Int(last[2].value()), 9)


def test_replace_dynamic_element() raises:
    var c = Coord(Int64(1), Int64(2))
    var r = c.replace[0](Int64(9))

    assert_equal(Int(r[0].value()), 9)
    assert_equal(Int(r[1].value()), 2)


def test_replace_static_with_static() raises:
    var c = Coord(ComptimeInt[3](), ComptimeInt[4]())
    var r = c.replace[1](ComptimeInt[9]())

    # Replacing a static with a static keeps the coord zero-sized.
    comptime assert size_of[type_of(r)]() == 0
    assert_equal(Int(r[1].value()), 9)


def test_replace_rank_one() raises:
    var c = Coord(ComptimeInt[5]())
    var r = c.replace[0](Int64(11))
    assert_equal(Int(r[0].value()), 11)


def test_replace_chains_for_multiple_dims() raises:
    var c = Coord(
        ComptimeInt[2](), ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]()
    )
    var r = c.replace[0](Int64(9)).replace[2](Int64(8))

    assert_equal(Int(r[0].value()), 9)
    assert_equal(Int(r[1].value()), 3)
    assert_equal(Int(r[2].value()), 8)
    assert_equal(Int(r[3].value()), 5)

    # Chaining costs only the dims actually replaced.
    comptime assert size_of[type_of(r)]() == 2 * size_of[Int64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
