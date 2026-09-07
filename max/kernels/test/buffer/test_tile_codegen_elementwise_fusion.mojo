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
"""Tests for the tile-based elementwise fusion contract.

Covers these platform-agnostic pieces:

- The real `Add.elementwise` / `Mul.elementwise` overloads on `TileTensor`
  (numeric checks on CPU).
- The `ElementwiseFusionTile` trait (compile-time conformance check).
- `get_kernel_tile_shape` (compile-time tile-shape lookup).
"""

from builtin_kernels.elementwise import Add, Mul
from extensibility import ElementwiseFusionTile, get_kernel_tile_shape

from layout import ComptimeInt, RowMajorLayout, TileTensor, row_major
from layout.tile_io import TileCopier
from layout.tile_layout import TensorLayout

from std.testing import TestSuite, assert_equal
from std.utils.index import IndexList

# 4x4 row-major float32 tile layout used by the numeric add test.
comptime _4x4 = RowMajorLayout[ComptimeInt[4], ComptimeInt[4]]


# ===-----------------------------------------------------------------------===#
# `Add.elementwise` on TileTensor (numeric, CPU-runnable)
# ===-----------------------------------------------------------------------===#


def test_tile_add_elementwise() raises:
    """`Add.elementwise` sums two tiles in place and returns the result."""
    var lhs_arr = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var rhs_arr = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(100 + i)
    )

    var lhs = TileTensor(lhs_arr, row_major[4, 4]())
    var rhs = TileTensor(rhs_arr, row_major[4, 4]())

    var result = Add.elementwise[.float32, _4x4](lhs, rhs)

    for i in range(4):
        for j in range(4):
            var expected = Float32((i * 4 + j) + (100 + i * 4 + j))
            # Returned tile holds the sum.
            assert_equal(result[i, j], expected)
            # The add is in place, so `lhs` is mutated too.
            assert_equal(lhs[i, j], expected)


def test_tile_mul_elementwise() raises:
    """`Mul.elementwise` multiplies two tiles in place and returns the result.
    """
    var lhs_arr = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    var rhs_arr = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(2 + i)
    )

    var lhs = TileTensor(lhs_arr, row_major[4, 4]())
    var rhs = TileTensor(rhs_arr, row_major[4, 4]())

    var result = Mul.elementwise[.float32, _4x4](lhs, rhs)

    for i in range(4):
        for j in range(4):
            var expected = Float32((i * 4 + j) * (2 + i * 4 + j))
            # Returned tile holds the product.
            assert_equal(result[i, j], expected)
            # The multiply is in place, so `lhs` is mutated too.
            assert_equal(lhs[i, j], expected)


# ===-----------------------------------------------------------------------===#
# `ElementwiseFusionTile` trait conformance (compile-time)
# ===-----------------------------------------------------------------------===#


struct _TileElemFusionProbe(ElementwiseFusionTile):
    """Minimal struct that conforms to `ElementwiseFusionTile`.

    Only exists to exercise trait conformance at compile time; `compute` is
    never actually invoked.
    """

    def __init__(out self):
        pass

    def compute[
        dtype: DType,
        rank: Int,
        LayoutType: TensorLayout,
        Copier: TileCopier,
    ](
        self,
        tile_coords: IndexList[rank],
        copier: Copier,
        dst: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        comptime assert False, "probe compute() is not callable"


def _requires_elementwise_fusion_tile[T: ElementwiseFusionTile]():
    pass


def test_elementwise_fusion_tile_conformance() raises:
    """The probe struct satisfies the `ElementwiseFusionTile` trait bound."""
    _requires_elementwise_fusion_tile[_TileElemFusionProbe]()


# ===-----------------------------------------------------------------------===#
# `get_kernel_tile_shape`
# ===-----------------------------------------------------------------------===#


def test_get_kernel_tile_shape() raises:
    """`get_kernel_tile_shape` returns a static 2D tile shape per target."""
    comptime gpu_shape = get_kernel_tile_shape[.float32, "gpu"]()
    assert_equal(gpu_shape[0], 16)
    assert_equal(gpu_shape[1], 16)

    comptime cpu_shape = get_kernel_tile_shape[.float32, "cpu"]()
    assert_equal(cpu_shape[0], 8)
    assert_equal(cpu_shape[1], 8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
