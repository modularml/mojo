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

from nn.attention.mha_mask import MASK_VALUE, ChunkedMask, TileMaskStatus
from std.testing import assert_equal

from std.utils.index import Index


def test_chunked_mask_status() raises:
    var mask = ChunkedMask[local_window_size=4]()

    assert_equal(
        mask.status(UInt32(0), Index(0, 0), Index(4, 4)),
        TileMaskStatus.NO_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(4, 4), Index(4, 4)),
        TileMaskStatus.NO_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(2, 2), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(0, 2), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(2, 0), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(0, 4), Index(4, 4)),
        TileMaskStatus.FULL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(4, 0), Index(4, 4)),
        TileMaskStatus.FULL_MASK,
    )

    # cases where tile_size >> local_window_size
    assert_equal(
        mask.status(UInt32(0), Index(100, 0), Index(128, 128)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(0, 0), Index(100, 100)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(50, 0), Index(100, 100)),
        TileMaskStatus.PARTIAL_MASK,
    )

    var bigger_mask = ChunkedMask[local_window_size=256]()
    assert_equal(
        bigger_mask.status(UInt32(0), Index(256, 256), Index(64, 128)),
        TileMaskStatus.NO_MASK,
    )
    assert_equal(
        bigger_mask.status(UInt32(0), Index(128, 0), Index(128, 128)),
        TileMaskStatus.NO_MASK,
    )


def test_chunked_mask_apply() raises:
    var mask = ChunkedMask[local_window_size=4]()

    var score_vec = SIMD[.float32, 4](0.0)
    score_vec[0] = 1.0
    score_vec[1] = 2.0
    score_vec[2] = 3.0
    score_vec[3] = 4.0

    comptime SIMD_T = SIMD[.float32, 4]
    var inf_vec = SIMD_T(MASK_VALUE)

    # first two dims should be arbitrary, we pass in junk just to help confirm.
    assert_equal(mask.mask(Index(0, 0, 0, 0), score_vec), score_vec)
    assert_equal(mask.mask(Index(10, 0, 4, 8), score_vec), inf_vec)
    assert_equal(mask.mask(Index(2, 10, 8, 8), score_vec), score_vec)

    assert_equal(
        mask.mask(Index(0, 4, 8, 10), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    assert_equal(
        mask.mask(Index(4, 0, 12, 10), score_vec),
        SIMD_T(MASK_VALUE, MASK_VALUE, 3.0, 4.0),
    )


def test_chunked_mask_bits() raises:
    """Test the 32-bit bitmask `mask_bits` returns after the BITMASK migration.

    Each bit `i` (0..31) corresponds to column `col_start + i`. Bit is 1 iff
    that column lives in the same `local_window_size`-wide chunk as
    `score_row` and is below `num_keys`.
    """
    # ---- W = 128 ----
    var mask128 = ChunkedMask[local_window_size=128]()

    # score_row=50 is in chunk [0, 128); batch [0, 32) entirely inside that
    # chunk; OOB irrelevant (num_keys=1000 >> col_start+32).
    assert_equal(
        mask128.mask_bits(UInt32(0), Int32(50), Int32(0), Int32(1000)),
        UInt32(0xFFFF_FFFF),
    )
    # score_row=50 chunk = [0, 128); batch [128, 160) entirely past chunk end.
    assert_equal(
        mask128.mask_bits(UInt32(0), Int32(50), Int32(128), Int32(1000)),
        UInt32(0),
    )
    # score_row=50 chunk = [0, 128); batch [112, 144) straddles chunk end at
    # global col 128 = bit index 16 within the batch.
    assert_equal(
        mask128.mask_bits(UInt32(0), Int32(50), Int32(112), Int32(1000)),
        UInt32((1 << 16) - 1),
    )
    # OOB clip: score_row=1000 chunk = [896, 1024); batch [992, 1024) inside
    # chunk; num_keys=1010 truncates to low 18 bits.
    assert_equal(
        mask128.mask_bits(UInt32(0), Int32(1000), Int32(992), Int32(1010)),
        UInt32((1 << 18) - 1),
    )

    # ---- W = 16 (chunk size < 32-bit batch width) ----
    var mask16 = ChunkedMask[local_window_size=16]()

    # score_row=50 chunk = [48, 64); batch [32, 64) covers bit indices
    # [16, 32) of the chunk.
    assert_equal(
        mask16.mask_bits(UInt32(0), Int32(50), Int32(32), Int32(1000)),
        UInt32(0xFFFF_0000),
    )
    # score_row=50 chunk = [48, 64); batch [0, 32) lies entirely below chunk.
    assert_equal(
        mask16.mask_bits(UInt32(0), Int32(50), Int32(0), Int32(1000)),
        UInt32(0),
    )


def main() raises:
    test_chunked_mask_status()
    test_chunked_mask_apply()
    test_chunked_mask_bits()
