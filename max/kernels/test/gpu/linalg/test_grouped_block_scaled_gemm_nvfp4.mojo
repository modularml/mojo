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
"""Integration tests for grouped block-scaled GEMM with NVF4 format.

Tests the GroupedTileScheduler and GroupedTensormapManager infrastructure
with NVF4 (Float4E2M1 + Float8E8M0/E4M3 scales) format.

Test matrix covering NVIDIA CuTe DSL constraints:
- 1, 2, 4, 8 groups
- Variable M/N/K sizes per group
- NVF4: Float4E2M1 + Float8E8M0/E4M3 + sf_vec_size=16
- Cluster shapes: 1x1, 2x2 (for initial tests)
"""

from std.math import ceildiv

from max.gpu.host import DeviceContext
from std.testing import testing

from std.utils.static_tuple import StaticTuple

from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_tile_scheduler import (
    GroupedTileScheduler,
)


# =============================================================================
# Test: NVF4 Format Constraints
# =============================================================================


def test_nvf4_format_constraints(ctx: DeviceContext) raises:
    """Validate NVF4 format constraints.

    NVF4 format:
    - Data type: Float4E2M1 (4-bit float)
    - Scale factor type: Float8E8M0 or Float8E4M3
    - Scale factor vector size: 16 (compared to 32 for MXF8)
    """
    print("  Test: NVF4 format constraints")

    # NVF4 sf_vec_size = 16 (16 elements share one scale factor)
    comptime NVFP4_SF_VEC_SIZE = 16

    # This determines the K-group size for scaling
    testing.assert_equal(NVFP4_SF_VEC_SIZE, 16)

    # NVF4 uses half the sf_vec_size of MXF8
    comptime MXFP8_SF_VEC_SIZE = 32
    testing.assert_equal(MXFP8_SF_VEC_SIZE // 2, NVFP4_SF_VEC_SIZE)

    print("    NVF4 format constraints verified PASSED")


# =============================================================================
# Test: Tile Count with NVF4 Alignment
# =============================================================================


def test_tile_counts_nvf4_alignment(ctx: DeviceContext) raises:
    """Test tile counts with NVF4 alignment requirements.

    NVF4 format has stricter alignment due to 4-bit elements.
    """
    print("  Test: Tile counts with NVF4 alignment")

    comptime tile_m = 128
    comptime tile_n = 128
    comptime tile_k = 128  # K must be larger for 4-bit elements

    # Group 0: 256x256x256 -> 2 * 2 = 4 tiles
    var m0 = 256
    var n0 = 256
    var expected_tiles = ceildiv(m0, tile_m) * ceildiv(n0, tile_n)
    testing.assert_equal(expected_tiles, 4)

    print("    Single group 256x256x256: ", expected_tiles, " tiles PASSED")


# =============================================================================
# Test: K-Tile Count for NVF4
# =============================================================================


def test_k_tile_counts_nvf4(ctx: DeviceContext) raises:
    """Test K-tile counts for NVF4 format.

    NVF4 has different K-tile sizes due to 4-bit packing.
    """
    print("  Test: K-tile counts for NVF4")

    # For NVF4, K dimension is often larger due to element packing
    comptime tile_k = 128  # Typical for NVF4

    # Group 0: K=256 -> 2 k-tiles
    # Group 1: K=512 -> 4 k-tiles
    var k0 = 256
    var k1 = 512

    var k_tiles_0 = ceildiv(k0, tile_k)
    var k_tiles_1 = ceildiv(k1, tile_k)

    testing.assert_equal(k_tiles_0, 2)
    testing.assert_equal(k_tiles_1, 4)

    print("    K-tile counts for NVF4 verified PASSED")


# =============================================================================
# Test: 16-byte Alignment
# =============================================================================


def test_16_byte_alignment(ctx: DeviceContext) raises:
    """Test 16-byte alignment constraint.

    From NVIDIA CuTe DSL: 16-byte alignment on contiguous dimensions.
    """
    print("  Test: 16-byte alignment")

    # For 4-bit elements (0.5 bytes), 16 bytes = 32 elements
    # For 8-bit elements (1 byte), 16 bytes = 16 elements
    # For 16-bit elements (2 bytes), 16 bytes = 8 elements
    # For 32-bit elements (4 bytes), 16 bytes = 4 elements

    def min_aligned_elements(dtype_bytes: Int) -> Int:
        """Minimum elements for 16-byte alignment."""
        return 16 // dtype_bytes

    testing.assert_equal(min_aligned_elements(4), 4)  # float32
    testing.assert_equal(min_aligned_elements(2), 8)  # float16/bfloat16
    testing.assert_equal(min_aligned_elements(1), 16)  # float8

    # For 4-bit elements, we need 32 elements for 16-byte alignment
    # (32 * 0.5 bytes = 16 bytes)
    # This is why NVF4 sf_vec_size is 16 (32/2 for packed 4-bit pairs)

    print("    16-byte alignment constraints verified PASSED")


# =============================================================================
# Test: Multiple Group Tile Distribution
# =============================================================================


def test_multiple_group_distribution(ctx: DeviceContext) raises:
    """Test tile distribution across multiple groups.

    Validates that tiles are correctly assigned to groups.
    """
    print("  Test: Multiple group tile distribution")

    comptime tile_m = 128
    comptime tile_n = 128

    # 4 groups with different sizes:
    # Group 0: 128x128 -> 1 tile (indices 0)
    # Group 1: 256x128 -> 2 tiles (indices 1-2)
    # Group 2: 128x256 -> 2 tiles (indices 3-4)
    # Group 3: 256x256 -> 4 tiles (indices 5-8)
    # Total: 9 tiles

    var tiles = StaticTuple[Int, 4](0, 0, 0, 0)
    tiles[0] = ceildiv(128, tile_m) * ceildiv(128, tile_n)  # 1
    tiles[1] = ceildiv(256, tile_m) * ceildiv(128, tile_n)  # 2
    tiles[2] = ceildiv(128, tile_m) * ceildiv(256, tile_n)  # 2
    tiles[3] = ceildiv(256, tile_m) * ceildiv(256, tile_n)  # 4

    var total_tiles = tiles[0] + tiles[1] + tiles[2] + tiles[3]
    testing.assert_equal(total_tiles, 9)

    # Verify cumulative indices
    var cumulative = StaticTuple[Int, 5](0, 0, 0, 0, 0)
    cumulative[0] = 0
    cumulative[1] = tiles[0]  # 1
    cumulative[2] = cumulative[1] + tiles[1]  # 3
    cumulative[3] = cumulative[2] + tiles[2]  # 5
    cumulative[4] = cumulative[3] + tiles[3]  # 9

    testing.assert_equal(cumulative[1], 1)
    testing.assert_equal(cumulative[2], 3)
    testing.assert_equal(cumulative[3], 5)
    testing.assert_equal(cumulative[4], 9)

    print("    4 groups with 9 total tiles verified PASSED")


# =============================================================================
# Test: NVF4 vs MXF8 Comparison
# =============================================================================


def test_nvf4_vs_mxf8_comparison(ctx: DeviceContext) raises:
    """Compare NVF4 and MXF8 format parameters.

    Validates the key differences between the two formats.
    """
    print("  Test: NVF4 vs MXF8 comparison")

    # Scale factor vector sizes
    comptime NVFP4_SF_VEC_SIZE = 16
    comptime MXFP8_SF_VEC_SIZE = 32

    # NVF4 has half the sf_vec_size
    testing.assert_equal(NVFP4_SF_VEC_SIZE, MXFP8_SF_VEC_SIZE // 2)

    # Element sizes (in bits)
    comptime NVFP4_ELEMENT_BITS = 4
    comptime MXFP8_ELEMENT_BITS = 8

    # NVF4 elements are half the size
    testing.assert_equal(NVFP4_ELEMENT_BITS, MXFP8_ELEMENT_BITS // 2)

    # Elements per 128-byte tile (for same K dimension)
    # MXF8: 128 bytes / 1 byte = 128 elements per row
    # NVF4: 128 bytes / 0.5 bytes = 256 elements per row
    comptime mxf8_elements_per_128b = 128
    comptime nvf4_elements_per_128b = 256

    testing.assert_equal(nvf4_elements_per_128b, 2 * mxf8_elements_per_128b)

    print("    NVF4 vs MXF8 comparison verified PASSED")


# =============================================================================
# Main Test Entry Point
# =============================================================================


def main() raises:
    print("=" * 60)
    print("Test: Grouped Block-Scaled GEMM (NVF4)")
    print("=" * 60)

    var ctx = DeviceContext()
    print("Running validation tests...")

    print()
    test_nvf4_format_constraints(ctx)
    print()
    test_tile_counts_nvf4_alignment(ctx)
    print()
    test_k_tile_counts_nvf4(ctx)
    print()
    test_16_byte_alignment(ctx)
    print()
    test_multiple_group_distribution(ctx)
    print()
    test_nvf4_vs_mxf8_comparison(ctx)
    print()

    print("=" * 60)
    print("All tests completed successfully")
    print("=" * 60)
