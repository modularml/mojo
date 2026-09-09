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
"""Equivalence of the typed tile layouts with their legacy counterparts.

Every `tile_layout_*_typed` must lower through `to_layout()` to exactly the
legacy `Layout` that the corresponding `tile_layout_*()` builds, so callers can
switch to the typed form without moving a byte in shared memory.

Layouts are compared through their printed form rather than `Layout.__eq__`:
`IntTuple` equality treats a single-element tuple as its scalar, which would
hide a change in the mode nesting.
"""

from std.sys import size_of
from std.testing import TestSuite, assert_equal

from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_k_major_typed,
    tile_layout_mn_major,
    tile_layout_mn_major_typed,
    tile_sf_layout_k_major,
    tile_sf_layout_k_major_typed,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle

# A K extent is tileable only when it spans a whole number of swizzle atoms;
# other combinations are rejected by the legacy layouts themselves.
comptime _k_fits_swizzle[
    dtype: DType, BK: Int, swizzle_mode: TensorMapSwizzle
] = (BK * size_of[dtype]()) % swizzle_mode.bytes() == 0


def _check_k_and_mn_major[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
]() raises:
    """Compares both major orders for one tile configuration.

    The MN-major layouts are the transpose of the K-major ones with the extents
    swapped, so a single legality check covers both.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        swizzle_mode: Memory access pattern swizzling mode.
    """
    comptime if _k_fits_swizzle[dtype, BK, swizzle_mode]:
        assert_equal(
            String(
                tile_layout_k_major_typed[
                    dtype, BM, BK, swizzle_mode
                ].to_layout()
            ),
            String(tile_layout_k_major[dtype, BM, BK, swizzle_mode]()),
        )
        assert_equal(
            String(
                tile_layout_mn_major_typed[
                    dtype, BK, BM, swizzle_mode
                ].to_layout()
            ),
            String(tile_layout_mn_major[dtype, BK, BM, swizzle_mode]()),
        )


def _check_all_swizzles[dtype: DType, BM: Int, BK: Int]() raises:
    """Runs `_check_k_and_mn_major` over every swizzle mode.

    Parameters:
        dtype: Element data type of the tensor.
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
    """
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_NONE]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_32B]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_64B]()
    _check_k_and_mn_major[dtype, BM, BK, TensorMapSwizzle.SWIZZLE_128B]()


def _check_all_shapes[dtype: DType]() raises:
    """Runs `_check_all_swizzles` over a range of tile shapes.

    `BM == 8` and `BK == 8` collapse an outer mode to extent 1, the case where
    the legacy layouts drop the stride to 0.

    Parameters:
        dtype: Element data type of the tensor.
    """
    _check_all_swizzles[dtype, 8, 8]()
    _check_all_swizzles[dtype, 8, 64]()
    _check_all_swizzles[dtype, 8, 128]()
    _check_all_swizzles[dtype, 64, 64]()
    _check_all_swizzles[dtype, 64, 128]()
    _check_all_swizzles[dtype, 128, 64]()
    _check_all_swizzles[dtype, 128, 256]()
    _check_all_swizzles[dtype, 256, 128]()


def _check_sf[BM: Int, BK: Int, SF_SCALE_SIZE: Int]() raises:
    """Compares the scale-factor layout for one tile configuration.

    Parameters:
        BM: Size of the M dimension in the tile.
        BK: Size of the K dimension in the tile.
        SF_SCALE_SIZE: Number of elements in a scale factor vector.
    """
    assert_equal(
        String(tile_sf_layout_k_major_typed[BM, BK, SF_SCALE_SIZE].to_layout()),
        String(tile_sf_layout_k_major[BM, BK, SF_SCALE_SIZE]()),
    )


def test_k_major_and_mn_major_match_legacy() raises:
    _check_all_shapes[.bfloat16]()
    _check_all_shapes[.float16]()
    _check_all_shapes[.float32]()
    _check_all_shapes[.float8_e4m3fn]()


def test_sf_layout_matches_legacy() raises:
    _check_sf[128, 64, 16]()
    _check_sf[128, 128, 16]()
    _check_sf[128, 128, 32]()
    _check_sf[128, 256, 32]()
    _check_sf[128, 512, 32]()
    _check_sf[256, 128, 16]()
    _check_sf[256, 256, 32]()
    _check_sf[256, 512, 32]()
    _check_sf[384, 192, 16]()
    _check_sf[512, 128, 32]()
    _check_sf[512, 256, 32]()
    _check_sf[640, 128, 32]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
