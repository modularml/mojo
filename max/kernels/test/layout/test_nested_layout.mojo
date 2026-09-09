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
"""Unit tests for nested (hierarchical) layouts in the parameterized
`Layout[shape_types, stride_types]` type, following CuTe Layout Algebra.

Reference: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md

CuTe invariants we enforce here:

- Shape and stride are congruent: a nested shape `((a,b),(c,d))` pairs with
  a nested stride `((s_a,s_b),(s_c,s_d))` of matching structure.
- Row-major over a nested shape is row-major over the flattened shape, then
  re-nested. For `((a,b),(c,d))`: flat shape `(a,b,c,d)`, flat strides
  `(b*c*d, c*d, d, 1)`, re-nested `((b*c*d, c*d), (d, 1))`.
- Column-major is the mirror: flat strides `(1, a, a*b, a*b*c)` re-nested
  to `((1, a), (a*b, a*b*c))`.
- `crd2idx` of a nested coord = dot product of matched leaves with matched
  strides. Hierarchical/exact, rank-matching, flat, and scalar coord forms
  all give the same offset (CuTe `crd2idx` invariants).
"""

from layout import Coord, Idx, TileTensor, row_major, col_major
from layout.int_tuple import IntTuple, coord_to_int_tuple
from layout.tile_layout import (
    Layout,
    blocked_product,
    col_major_nested,
    row_major_nested,
)
from std.testing import assert_equal, assert_true, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ===----------------------------------------------------------------------=== #
# Group A — Coord helper predicates on nested shapes
# (Currently broken: see /tmp/test_softmax_nested_layout.mojo for the failing
#  cases. These tests pin the expected post-fix behavior.)
# ===----------------------------------------------------------------------=== #


def test_nested_coord_flat_rank() raises:
    """Pins `flat_rank` walking nesting on a `Coord(Coord, Coord)` shape."""
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    comptime ShapeT = type_of(s)
    assert_equal(ShapeT.rank, 2)
    assert_equal(ShapeT.flat_rank, 4)
    _ = s


def test_nested_coord_all_dims_known_static() raises:
    """Pins that a fully-static nested Coord reports `all_dims_known=True`."""
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    comptime ShapeT = type_of(s)
    assert_true(ShapeT.all_dims_known)
    _ = s


def test_nested_coord_static_product() raises:
    """Pins recursive `static_product`: 4*16*1*1 = 64, 2*3*4*5 = 120."""
    var s1 = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    var s2 = Coord(Coord(Idx[2], Idx[3]), Coord(Idx[4], Idx[5]))
    var s3 = Coord(Coord(Coord(Idx[2], Idx[3]), Idx[4]), Idx[5])  # depth 2
    assert_equal(type_of(s1).static_product, 64)
    assert_equal(type_of(s2).static_product, 120)
    assert_equal(type_of(s3).static_product, 120)
    _ = s1
    _ = s2
    _ = s3


def test_nested_coord_product_runtime() raises:
    """Pins `.product()` (runtime) recursing through nested Coord leaves."""
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    assert_equal(Int(s.product()), 64)


# ===----------------------------------------------------------------------=== #
# Group B — row_major / col_major on nested shapes
# ===----------------------------------------------------------------------=== #


def test_row_major_nested_value_form_basic() raises:
    """Builds a nested `row_major` and pins flat/nested stride agreement.

    Flat shape `(4,16,1,1)` -> flat strides `(16,1,1,1)` -> nested
    `((16,1),(1,1))`.
    """
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    var L = row_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT.rank, 2)
    assert_equal(LT.flat_rank, 4)
    # Shape leaves
    assert_equal(LT._shape_types[0].ParamListType[0].static_value, 4)
    assert_equal(LT._shape_types[0].ParamListType[1].static_value, 16)
    assert_equal(LT._shape_types[1].ParamListType[0].static_value, 1)
    assert_equal(LT._shape_types[1].ParamListType[1].static_value, 1)
    # Stride leaves
    assert_equal(LT._stride_types[0].ParamListType[0].static_value, 16)
    assert_equal(LT._stride_types[0].ParamListType[1].static_value, 1)
    assert_equal(LT._stride_types[1].ParamListType[0].static_value, 1)
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 1)
    _ = L


def test_row_major_nested_value_form_asymmetric() raises:
    """Pins asymmetric nested row-major.

    Shape `((4,16),(8,2))` -> flat `(4,16,8,2)` -> flat strides
    `(256,16,2,1)` -> nested `((256,16),(2,1))`.
    """
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[8], Idx[2]))
    var L = row_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT._stride_types[0].ParamListType[0].static_value, 256)
    assert_equal(LT._stride_types[0].ParamListType[1].static_value, 16)
    assert_equal(LT._stride_types[1].ParamListType[0].static_value, 2)
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 1)
    _ = L


def test_col_major_nested_value_form() raises:
    """Pins col-major on a nested shape.

    `((4,16),(8,2))` -> flat strides `(1,4,64,512)` -> nested
    `((1,4),(64,512))`.
    """
    var s = Coord(Coord(Idx[4], Idx[16]), Coord(Idx[8], Idx[2]))
    var L = col_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT._stride_types[0].ParamListType[0].static_value, 1)
    assert_equal(LT._stride_types[0].ParamListType[1].static_value, 4)
    assert_equal(LT._stride_types[1].ParamListType[0].static_value, 64)
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 512)
    _ = L


def test_row_major_nested_vs_blocked_product_different_strides() raises:
    """Pins the stride divergence between `row_major(nested)` and
    `blocked_product` on the same nested shape.

    Both share shape `((B0,T0),(B1,T1))` but assign different stride
    conventions:

    - `row_major(((B0,T0),(B1,T1)))` is row-major over the flattened
      shape `(B0,T0,B1,T1)`, re-nested. The FIRST sub of each mode
      (`B0`/`B1`) is the OUTER (coarse) sub.
    - `blocked_product(block=row_major[B0,B1], tiler=row_major[T0,T1])`
      treats the FIRST sub of each mode as the BLOCK (fine, contiguous)
      and the SECOND sub as the TILER (coarse).
    """
    comptime B0 = 2
    comptime B1 = 3
    comptime T0 = 4
    comptime T1 = 5
    var bp = blocked_product(row_major[B0, B1](), row_major[T0, T1]())
    var rm = row_major_nested(
        Coord(Coord(Idx[B0], Idx[T0]), Coord(Idx[B1], Idx[T1]))
    )
    var c = Coord(Coord(Idx[1], Idx[2]), Coord(Idx[1], Idx[3]))
    # blocked_product: mode 0 stride = (block_stride[0]=3, cosize*tiler_stride[0]=6*5=30),
    #                  mode 1 stride = (block_stride[1]=1, cosize*tiler_stride[1]=6*1=6).
    # Offset = 1*3 + 2*30 + 1*1 + 3*6 = 82.
    assert_equal(bp(c), 82)
    # row_major nested: flat shape (B0,T0,B1,T1) = (2,4,3,5), flat strides
    # (60, 15, 5, 1), nested ((60, 15), (5, 1)).
    # Offset = 1*60 + 2*15 + 1*5 + 3*1 = 98.
    assert_equal(rm(c), 98)
    var zero = Coord(Coord(Idx[0], Idx[0]), Coord(Idx[0], Idx[0]))
    assert_equal(bp(zero), 0)
    assert_equal(rm(zero), 0)


# ===----------------------------------------------------------------------=== #
# Group C — Layout.__call__ on the nested row_major layout
# (CuTe crd2idx — exact/rank/flat/scalar coord forms agree)
# ===----------------------------------------------------------------------=== #


def test_layout_call_row_major_nested_exact() raises:
    """Layout((4,16),(1,1)) row-major -> strides ((16,1),(1,1)). Exact-match
    coord ((2,5),(0,0)) -> 2*16 + 5*1 + 0*1 + 0*1 = 37."""
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    )
    var c = Coord(Coord(Idx[2], Idx[5]), Coord(Idx[0], Idx[0]))
    assert_equal(L(c), 37)


def test_layout_call_row_major_nested_flat() raises:
    """Flat coord (2,5,0,0) -> same offset 37 via flat dot product."""
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    )
    assert_equal(
        L(Coord(Idx[2], Idx[5], Idx[0], Idx[0])),
        37,
    )


def test_layout_call_row_major_nested_consistency_asymmetric() raises:
    """For shape ((4,16),(8,2)) strides ((256,16),(2,1)), check that exact /
    flat coord forms give the same offset on three coords."""
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[8], Idx[2]))
    )
    # ((1,2),(3,1)) -> 1*256 + 2*16 + 3*2 + 1*1 = 295
    var ce = Coord(Coord(Idx[1], Idx[2]), Coord(Idx[3], Idx[1]))
    var cf = Coord(Idx[1], Idx[2], Idx[3], Idx[1])
    assert_equal(L(ce), 295)
    assert_equal(L(cf), 295)
    # ((0,0),(0,0)) -> 0
    assert_equal(
        L(Coord(Coord(Idx[0], Idx[0]), Coord(Idx[0], Idx[0]))),
        0,
    )
    # Top corner ((3,15),(7,1)) -> 3*256 + 15*16 + 7*2 + 1*1 = 1023
    var top = Coord(Coord(Idx[3], Idx[15]), Coord(Idx[7], Idx[1]))
    assert_equal(L(top), 1023)


# ===----------------------------------------------------------------------=== #
# Group D — TileTensor on a nested row_major layout
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_dim_on_nested_returns_product() raises:
    """TileTensor.dim[i]() must return the product of the i-th outer-mode
    sub-shape, not trap. For shape ((4,16),(1,1)): dim[0]=64, dim[1]=1."""
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    )
    var storage = Array[Float32, 64](fill=0.0)
    var t = TileTensor(storage, L)
    assert_equal(Int(t.dim[0]()), 64)
    assert_equal(Int(t.dim[1]()), 1)


def test_tile_tensor_load_store_via_nested_coord() raises:
    """Round-trip a value with hierarchical and flat coord forms."""
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[1], Idx[1]))
    )
    var storage = Array[Float32, 64](fill=0.0)
    var t = TileTensor(storage, L)
    # Hierarchical write at ((2,5),(0,0)) -> offset 37.
    t[Coord(Coord(Idx[2], Idx[5]), Coord(Idx[0], Idx[0]))] = 42.0
    assert_equal(storage[37], 42.0)
    # Flat read agrees.
    assert_equal(t[2, 5, 0, 0], 42.0)


def test_tile_tensor_tile_outer_mode_basic() raises:
    """`.tile[inner_h, inner_w](outer_h, outer_w)` slices ONE outer index
    per mode, returning a rank-2 sub-tile whose shape is the inner sub-
    shape. For ((OUTER_H=4, FRAG_H=16), (OUTER_W=1, FRAG_W=1)):
    `.tile[16, 1](i, j)` returns a (16, 1) view at outer index (i, j).
    """
    comptime OUTER_H = 4
    comptime FRAG_H = 16
    comptime OUTER_W = 1
    comptime FRAG_W = 1
    var L = row_major_nested(
        Coord(
            Coord(Idx[OUTER_H], Idx[FRAG_H]), Coord(Idx[OUTER_W], Idx[FRAG_W])
        )
    )
    var storage = Array[Float32, 64](fill=0.0)
    var t = TileTensor(storage, L)
    # Mark a single element via a flat write so we can verify the tile.
    # Outer (i=2, j=0), inner (frag_h=5, frag_w=0): flat (2,5,0,0) -> offset 37.
    t[2, 5, 0, 0] = 99.0
    var sub = t.tile[FRAG_H, FRAG_W](2, 0)
    # sub is rank-2 (FRAG_H, FRAG_W) = (16, 1). Element (5, 0) holds 99.
    assert_equal(sub[5, 0], 99.0)
    # Other tiles (i=0, j=0) should not be touched.
    var sub_zero = t.tile[FRAG_H, FRAG_W](0, 0)
    assert_equal(sub_zero[5, 0], 0.0)


def test_tile_tensor_tile_outer_mode_asymmetric() raises:
    """`.tile[inner_h, inner_w](i, j)` on shape ((B0,T0),(B1,T1)) returns
    the (T0, T1) sub-tile at outer index (i, j) — depth-1 nested case.
    """
    comptime B0 = 2
    comptime B1 = 3
    comptime T0 = 4
    comptime T1 = 5
    var L = row_major_nested(
        Coord(Coord(Idx[B0], Idx[T0]), Coord(Idx[B1], Idx[T1]))
    )
    var storage = Array[Float32, B0 * B1 * T0 * T1](fill=0.0)
    var t = TileTensor(storage, L)
    # Mark one element: outer (1, 2), inner (3, 1).
    t[Coord(Coord(Idx[1], Idx[3]), Coord(Idx[2], Idx[1]))] = 77.0
    var sub = t.tile[T0, T1](1, 2)
    assert_equal(sub[3, 1], 77.0)


def test_tile_tensor_tile_coord_form() raises:
    """`.tile[*sizes](coordinates: Coord)` is the Coord-arg overload
    of `.tile[]`; same semantics as the variadic-`Int` form.
    """
    comptime B0 = 2
    comptime B1 = 3
    comptime T0 = 4
    comptime T1 = 5
    var L = row_major_nested(
        Coord(Coord(Idx[B0], Idx[T0]), Coord(Idx[B1], Idx[T1]))
    )
    var storage = Array[Float32, B0 * B1 * T0 * T1](fill=0.0)
    var t = TileTensor(storage, L)
    t[Coord(Coord(Idx[1], Idx[3]), Coord(Idx[2], Idx[1]))] = 77.0
    var sub = t.tile[T0, T1](Coord(Idx[1], Idx[2]))
    assert_equal(sub[3, 1], 77.0)


def test_tile_tensor_tile_on_col_major() raises:
    """`.tile[]` on a `col_major_nested` parent. For shape
    `((B0,T0),(B1,T1))` col-major has flat strides `(1, B0, B0*T0,
    B0*T0*B1)` re-nested to `((1, B0), (B0*T0, B0*T0*B1))`. The
    innermost sub-strides become the sub-tile's strides.
    """
    comptime B0 = 2
    comptime B1 = 3
    comptime T0 = 4
    comptime T1 = 5
    var L = col_major_nested(
        Coord(Coord(Idx[B0], Idx[T0]), Coord(Idx[B1], Idx[T1]))
    )
    var storage = Array[Float32, B0 * B1 * T0 * T1](fill=0.0)
    var t = TileTensor(storage, L)
    # Mark via hierarchical coord at outer (1, 2), inner (3, 1).
    # Offset = 1*1 + 3*B0 + 2*(B0*T0) + 1*(B0*T0*B1) = 1+6+16+24 = 47.
    t[Coord(Coord(Idx[1], Idx[3]), Coord(Idx[2], Idx[1]))] = 88.0
    assert_equal(storage[47], 88.0)
    var sub = t.tile[T0, T1](1, 2)
    assert_equal(sub[3, 1], 88.0)


def test_tile_tensor_tile_on_flat_parent() raises:
    """`.tile[]` works on a flat (non-nested) parent too — falls
    back to classic `coord * tile_size * stride` offset per mode.

    For row_major[4, 8] with `.tile[2, 4](1, 0)`:
    offset = 1*2*8 + 0*4*1 = 16. Sub-tile starts at row 2, col 0.
    """
    var L = row_major[4, 8]()
    var storage = Array[Float32, 32](fill=0.0)
    var t = TileTensor(storage, L)
    # Mark row 2, col 3 = offset 2*8 + 3 = 19.
    t[2, 3] = 55.0
    # `.tile[2, 4](1, 0)` should give the (2, 4) sub-tile at
    # rows 2..4, cols 0..4. Element (0, 3) of sub-tile = row 2 col 3.
    var sub = t.tile[2, 4](1, 0)
    assert_equal(sub[0, 3], 55.0)


# ===----------------------------------------------------------------------=== #
# Group E — Production-shape geometry from MhaMmaOp.ATT_LAYOUT_NESTED.
# These tests pin the SAME shape/strides that
# `max/kernels/src/nn/attention/gpu/amd_structured/mha_mma_op.mojo`'s
# `ATT_LAYOUT_NESTED` alias produces for the default MHA kernel config
# (KV_BLOCK=128, Q_BLOCK_SIZE=32, MMA_M=MMA_N=32, MMA_K=16). We inline
# the constants here (rather than importing `MhaMmaOp`) so the test stays
# in the layout package and doesn't pull a cross-package dep on
# `//max:nn` into `//max/kernels/test/layout:*`. The production alias is
# verified by virtue of being built (`./bazelw build //max:nn`).
# ===----------------------------------------------------------------------=== #


def test_mha_mma_op_att_layout_nested_geometry() raises:
    """Pins shape/stride of MhaMmaOp.ATT_LAYOUT_NESTED for the default
    MHA kernel config — `((4, 16), (1, 1))` shape, `((16, 1), (1, 1))`
    strides, same per-lane 64-FP32 storage as the flat ATT_LAYOUT.
    """
    # ATT_LAYOUT_NESTED-equivalent geometry. Source of truth lives at
    # max/kernels/src/nn/attention/gpu/amd_structured/mha_mma_op.mojo:
    # ATT_LAYOUT_NESTED.
    comptime KV_BLOCK = 128
    comptime Q_BLOCK_SIZE = 32
    comptime MMA_M = 32
    comptime MMA_N = 32
    comptime H = KV_BLOCK // MMA_M  # 4
    comptime W = Q_BLOCK_SIZE // MMA_N  # 1
    comptime FRAG_H_COL_L = 16
    comptime FRAG_W_COL_L = 1
    var L = row_major_nested(
        Coord(
            Coord(Idx[H], Idx[FRAG_H_COL_L]),
            Coord(Idx[W], Idx[FRAG_W_COL_L]),
        )
    )
    comptime ATT_T = type_of(L)

    # Outer rank 2, flat_rank 4, fully static.
    assert_equal(ATT_T.rank, 2)
    assert_equal(ATT_T.flat_rank, 4)
    assert_true(ATT_T.shape_known)
    # Total per-lane storage matches the flat form: 4*16*1*1 = 64.
    assert_equal(ATT_T.static_product, 64)
    # Shape leaves: ((H, FRAG_H), (W, FRAG_W)) = ((4, 16), (1, 1)).
    assert_equal(ATT_T._shape_types[0].ParamListType[0].static_value, 4)
    assert_equal(ATT_T._shape_types[0].ParamListType[1].static_value, 16)
    assert_equal(ATT_T._shape_types[1].ParamListType[0].static_value, 1)
    assert_equal(ATT_T._shape_types[1].ParamListType[1].static_value, 1)
    # Row-major-over-flat strides re-nested: ((16, 1), (1, 1)).
    assert_equal(ATT_T._stride_types[0].ParamListType[0].static_value, 16)
    assert_equal(ATT_T._stride_types[0].ParamListType[1].static_value, 1)
    assert_equal(ATT_T._stride_types[1].ParamListType[0].static_value, 1)
    assert_equal(ATT_T._stride_types[1].ParamListType[1].static_value, 1)
    _ = L


def test_mha_mma_op_att_layout_nested_tile_semantics() raises:
    """Drives the softmax `.tile[FRAG_H, FRAG_W](outer_h, outer_w)`
    access pattern on the ATT_LAYOUT_NESTED-equivalent geometry — host-
    side stand-in for what `_col_reduce_at_j` would do on GPU under the
    nested layout.
    """
    comptime KV_BLOCK = 128
    comptime Q_BLOCK_SIZE = 32
    comptime MMA_M = 32
    comptime MMA_N = 32
    comptime H = KV_BLOCK // MMA_M  # 4
    comptime W = Q_BLOCK_SIZE // MMA_N  # 1
    comptime FRAG_H_COL_L = 16
    comptime FRAG_W_COL_L = 1
    var L = row_major_nested(
        Coord(
            Coord(Idx[H], Idx[FRAG_H_COL_L]),
            Coord(Idx[W], Idx[FRAG_W_COL_L]),
        )
    )
    var storage = Array[Float32, 64](fill=0.0)
    var att = TileTensor(storage, L)

    # Walk the (H=4, W=1) outer grid via `.tile[FRAG_H, FRAG_W](i, j)`
    # and stamp a value into the (i, 0)-th base tile's first per-lane
    # row.
    comptime for outer_h in range(H):
        var base = att.tile[FRAG_H_COL_L, FRAG_W_COL_L](outer_h, 0)
        base[0, 0] = Float32(100 + outer_h)

    # Row-major over (H, FRAG_H, W, FRAG_W) = (4, 16, 1, 1) places
    # (outer_h, 0, 0, 0) at offset outer_h * 16 in the contiguous
    # storage.
    assert_equal(storage[0], 100.0)
    assert_equal(storage[16], 101.0)
    assert_equal(storage[32], 102.0)
    assert_equal(storage[48], 103.0)


# ===----------------------------------------------------------------------=== #
# Group F — Deeper nesting (depth-2 and depth-3 shapes).
# `row_major_nested` / `col_major_nested` re-nest the flat strides into
# whatever tree the input shape has; these tests pin shape/stride pairs
# for shapes with mixed scalar/tuple subtrees and for fully-recursive
# shapes up to depth-3.
# ===----------------------------------------------------------------------=== #


def test_row_major_nested_depth2_mixed_subtree() raises:
    """Shape `((a, (b, c)), (d, e))` mixes a scalar and a sub-tuple inside
    one outer mode.

    For a=2, b=3, c=4, d=5, e=7: flat (2,3,4,5,7) → row-major flat strides
    (420, 140, 35, 7, 1) → re-nested `((420, (140, 35)), (7, 1))`.
    """
    var s = Coord(
        Coord(Idx[2], Coord(Idx[3], Idx[4])),
        Coord(Idx[5], Idx[7]),
    )
    var L = row_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT.rank, 2)
    assert_equal(LT.flat_rank, 5)
    # Outer mode 0: stride[0] = 420 (scalar leaf), stride[1] = (140, 35) (tuple).
    assert_equal(LT._stride_types[0].ParamListType[0].static_value, 420)
    assert_equal(
        LT._stride_types[0].ParamListType[1].ParamListType[0].static_value, 140
    )
    assert_equal(
        LT._stride_types[0].ParamListType[1].ParamListType[1].static_value, 35
    )
    # Outer mode 1: stride leaves (7, 1).
    assert_equal(LT._stride_types[1].ParamListType[0].static_value, 7)
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 1)
    # Hierarchical and flat coord forms agree on offset.
    var ch = Coord(
        Coord(Idx[1], Coord(Idx[2], Idx[3])),
        Coord(Idx[4], Idx[5]),
    )
    var cf = Coord(Idx[1], Idx[2], Idx[3], Idx[4], Idx[5])
    # Offset = 1*420 + 2*140 + 3*35 + 4*7 + 5*1 = 838.
    assert_equal(L(ch), 838)
    assert_equal(L(cf), 838)


def test_col_major_nested_depth2_mixed_subtree() raises:
    """Col-major mirror of the above.

    For a=2, b=3, c=4, d=5, e=7: flat (2,3,4,5,7) → col-major flat strides
    (1, 2, 6, 24, 120) → re-nested `((1, (2, 6)), (24, 120))`.
    """
    var s = Coord(
        Coord(Idx[2], Coord(Idx[3], Idx[4])),
        Coord(Idx[5], Idx[7]),
    )
    var L = col_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT._stride_types[0].ParamListType[0].static_value, 1)
    assert_equal(
        LT._stride_types[0].ParamListType[1].ParamListType[0].static_value, 2
    )
    assert_equal(
        LT._stride_types[0].ParamListType[1].ParamListType[1].static_value, 6
    )
    assert_equal(LT._stride_types[1].ParamListType[0].static_value, 24)
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 120)
    _ = L


def test_row_major_nested_depth2_symmetric() raises:
    """Fully-symmetric depth-2 shape `(((a, b), c), ((d, e), f))`.

    For a=2, b=3, c=4, d=5, e=6, f=7: flat (2,3,4,5,6,7) → row-major flat
    strides (2520, 840, 210, 42, 7, 1) → re-nested
    `(((2520, 840), 210), ((42, 7), 1))`.
    """
    var s = Coord(
        Coord(Coord(Idx[2], Idx[3]), Idx[4]),
        Coord(Coord(Idx[5], Idx[6]), Idx[7]),
    )
    var L = row_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT.flat_rank, 6)
    # First outer mode: stride[0][0] = (2520, 840), stride[0][1] = 210.
    assert_equal(
        LT._stride_types[0].ParamListType[0].ParamListType[0].static_value, 2520
    )
    assert_equal(
        LT._stride_types[0].ParamListType[0].ParamListType[1].static_value, 840
    )
    assert_equal(LT._stride_types[0].ParamListType[1].static_value, 210)
    # Second outer mode: stride[1][0] = (42, 7), stride[1][1] = 1.
    assert_equal(
        LT._stride_types[1].ParamListType[0].ParamListType[0].static_value, 42
    )
    assert_equal(
        LT._stride_types[1].ParamListType[0].ParamListType[1].static_value, 7
    )
    assert_equal(LT._stride_types[1].ParamListType[1].static_value, 1)
    _ = L


def test_row_major_nested_depth3() raises:
    """Depth-3 shape `((((a, b), c), d), e)`.

    For a=2, b=3, c=4, d=5, e=7: flat (2,3,4,5,7) → flat strides
    (420, 140, 35, 7, 1) → re-nested `((((420, 140), 35), 7), 1)`.
    """
    var s = Coord(
        Coord(Coord(Coord(Idx[2], Idx[3]), Idx[4]), Idx[5]),
        Idx[7],
    )
    var L = row_major_nested(s)
    comptime LT = type_of(L)
    assert_equal(LT.flat_rank, 5)
    # stride[0] is depth-3 nested; stride[1] is a flat leaf.
    comptime s0 = LT._stride_types[0].ParamListType
    comptime s00 = s0[0].ParamListType
    comptime s000 = s00[0].ParamListType
    assert_equal(s000[0].static_value, 420)
    assert_equal(s000[1].static_value, 140)
    assert_equal(s00[1].static_value, 35)
    assert_equal(s0[1].static_value, 7)
    assert_equal(LT._stride_types[1].static_value, 1)
    # Hierarchical/flat coord agreement.
    var ch = Coord(
        Coord(Coord(Coord(Idx[1], Idx[2]), Idx[3]), Idx[4]),
        Idx[6],
    )
    var cf = Coord(Idx[1], Idx[2], Idx[3], Idx[4], Idx[6])
    # Offset = 1*420 + 2*140 + 3*35 + 4*7 + 6*1 = 839.
    assert_equal(L(ch), 839)
    assert_equal(L(cf), 839)


# ===----------------------------------------------------------------------=== #
# Group G — the nested-layout LayoutTensor bridge
# ===----------------------------------------------------------------------=== #


def test_coord_to_int_tuple_types_nested() raises:
    """The type-only `coord_to_int_tuple` recurses into nested `Coord`s.

    Every level must shrink the pack it recurses on, otherwise the comptime
    call never terminates. `((2, (3, 5)), 4)` mixes leaf and nested siblings
    at both levels.
    """
    var s = Coord(Coord(Idx[2], Coord(Idx[3], Idx[5])), Idx[4])
    comptime t = coord_to_int_tuple[*type_of(s).element_types]()
    assert_equal(t, IntTuple(IntTuple(2, IntTuple(3, 5)), 4))
    _ = s


def test_to_layout_tensor_nested_layout() raises:
    """`to_layout_tensor()` carries a nested layout across unchanged.

    The runtime shape/stride are flat arrays of the layout's leaves, so a
    rank-2 nested layout contributes four entries, not two.
    """
    var L = row_major_nested(
        Coord(Coord(Idx[4], Idx[16]), Coord(Idx[2], Idx[8]))
    )
    var storage = Array[Float32, 1024](fill=0.0)
    var t = TileTensor(storage, L)
    var lt = t.to_layout_tensor()

    comptime LTLayout = type_of(lt).layout
    assert_equal(LTLayout.shape, IntTuple(IntTuple(4, 16), IntTuple(2, 8)))

    # Flat shape (4, 16, 2, 8) -> flat strides (256, 16, 8, 1).
    var shape = lt.runtime_layout.shape.value
    var stride = lt.runtime_layout.stride.value
    assert_equal(shape, type_of(shape)(4, 16, 2, 8))
    assert_equal(stride, type_of(stride)(256, 16, 8, 1))
