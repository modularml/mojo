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

from layout import Layout, LayoutTensor
from nn.attention.mha_mask import (
    MASK_VALUE,
    CausalPaddingMask,
    MaskStrategy,
    TileMaskStatus,
)
from std.testing import assert_equal

from std.utils.index import Index


def test_causal_padding_mask_status() raises:
    """Test tile status classification for CausalPaddingMask.

    With the set split (commit 2), `status` is now precise about both the
    causal diagonal AND the per-sequence padding boundary.
    """
    var storage: Array[UInt32, _] = [6, 4]  # valid_length per sequence
    var valid_lengths = LayoutTensor[.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    # ---- Causal FULL_MASK cases (independent of padding). ----
    # Tile entirely above diagonal: q < k always.
    assert_equal(
        mask.status(UInt32(0), Index(0, 4), Index(4, 4)),
        TileMaskStatus.FULL_MASK,
    )
    assert_equal(
        mask.status(UInt32(0), Index(0, 4), Index(2, 4)),
        TileMaskStatus.FULL_MASK,
    )

    # ---- Causal NO_MASK + padding doesn't cut -> NO_MASK. ----
    # seq 0 (valid_len=6): tile k range [0,4), all within valid_len.
    assert_equal(
        mask.status(UInt32(0), Index(4, 0), Index(4, 4)),
        TileMaskStatus.NO_MASK,
    )
    # seq 1 (valid_len=4): tile k range [0,4), exactly at the boundary
    # (k=3 is the last visible position, tile end = 4 <= valid_len = 4).
    assert_equal(
        mask.status(UInt32(1), Index(4, 0), Index(4, 4)),
        TileMaskStatus.NO_MASK,
    )

    # ---- Causal PARTIAL_MASK + padding doesn't cut -> PARTIAL_MASK. ----
    assert_equal(
        mask.status(UInt32(0), Index(2, 2), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )

    # ---- Padding-driven FULL_MASK: first column already past valid_len. ----
    # seq 1 (valid_len=4): tile starts at k=4, padding-masked.
    assert_equal(
        mask.status(UInt32(1), Index(4, 4), Index(4, 4)),
        TileMaskStatus.FULL_MASK,
    )
    # seq 0 (valid_len=6): tile starts at k=8, padding-masked.
    assert_equal(
        mask.status(UInt32(0), Index(8, 8), Index(4, 4)),
        TileMaskStatus.FULL_MASK,
    )

    # ---- Padding cuts mid-tile -> PARTIAL_MASK. ----
    # seq 0 (valid_len=6): tile (q=[8,12), k=[4,8)). Causal alone is NO_MASK
    # (8 >= 7) but padding boundary at k=6 cuts the tile.
    assert_equal(
        mask.status(UInt32(0), Index(8, 4), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )
    # seq 1 (valid_len=4): tile (q=[4,8), k=[2,6)). Causal NO_MASK; padding
    # at k=4 cuts.
    assert_equal(
        mask.status(UInt32(1), Index(4, 2), Index(4, 4)),
        TileMaskStatus.PARTIAL_MASK,
    )


def test_causal_padding_mask_set_split() raises:
    """Test the {NO_MASK, PARTIAL_MASK} partition exposed by
    `masked_set_ends` / `nonfull_sets` / `mask_strategies`.
    """
    var storage: Array[UInt32, _] = [10, 6]  # valid_length per sequence
    var valid_lengths = LayoutTensor[.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    # count_nonfull_sets = 2 (no more UNKNOWN_MASK).
    comptime assert mask.count_nonfull_sets(1, 4) == 2

    comptime nonfull = mask.nonfull_sets[BM=1, BN=4]()
    comptime assert nonfull[0] == TileMaskStatus.NO_MASK
    comptime assert nonfull[1] == TileMaskStatus.PARTIAL_MASK

    comptime strategies = mask.mask_strategies[BM=1, BN=4]()
    comptime assert strategies[0] == MaskStrategy.NO_MASK
    comptime assert strategies[1] == MaskStrategy.BITMASK

    # ---- seq 0 (valid_len=10), BM=1, BN=4, page_size=1 ----
    # row=12: visible k in [0, 10). Tile 0=[0,4) full, tile 1=[4,8) full,
    # tile 2=[8,12) partial (k=10,11 padding-masked), tile 3+ fully masked.
    # num_unmasked = min(13, 10) // 4 = 10 // 4 = 2.
    # total_iters = ceildiv(min(13, 10), 4) = ceildiv(10, 4) = 3.
    var ends_a = mask.masked_set_ends[BM=1, BN=4, page_size=1](
        UInt32(0), UInt32(12), UInt32(64)
    )
    assert_equal(ends_a[0], UInt32(2))
    assert_equal(ends_a[1], UInt32(3))
    assert_equal(
        mask.total_iters[BM=1, BN=4, page_size=1](
            UInt32(0), UInt32(12), UInt32(64)
        ),
        UInt32(3),
    )

    # row=0: visible k in [0, 1). Tile 0=[0,4) partial (causal cut), tile 1+
    # fully masked. num_unmasked = min(1, 10) // 4 = 0. total = ceildiv(1, 4) = 1.
    var ends_b = mask.masked_set_ends[BM=1, BN=4, page_size=1](
        UInt32(0), UInt32(0), UInt32(64)
    )
    assert_equal(ends_b[0], UInt32(0))
    assert_equal(ends_b[1], UInt32(1))

    # ---- seq 1 (valid_len=6), BM=1, BN=4, page_size=1 ----
    # row=20: visible k in [0, 6). Tile 0=[0,4) full unmasked,
    # tile 1=[4,8) partial (padding-cut), tile 2+ fully masked.
    # num_unmasked = min(21, 6) // 4 = 6 // 4 = 1.
    # total_iters = ceildiv(min(21, 6), 4) = ceildiv(6, 4) = 2.
    var ends_c = mask.masked_set_ends[BM=1, BN=4, page_size=1](
        UInt32(1), UInt32(20), UInt32(64)
    )
    assert_equal(ends_c[0], UInt32(1))
    assert_equal(ends_c[1], UInt32(2))


def test_causal_padding_mask_apply() raises:
    """Test per-element masking for CausalPaddingMask."""
    var storage: Array[UInt32, _] = [3]  # valid_length for seq 0
    var valid_lengths = LayoutTensor[.uint32, Layout.row_major(1)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    var score_vec = SIMD[.float32, 4](0.0)
    score_vec[0] = 1.0
    score_vec[1] = 2.0
    score_vec[2] = 3.0
    score_vec[3] = 4.0

    comptime SIMD_T = SIMD[.float32, 4]
    var inf_vec = SIMD_T(MASK_VALUE)

    # q=4, k=0..3: causal allows all (q >= k), padding allows k<3.
    # k=0: visible, k=1: visible, k=2: visible, k=3: padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 4, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, MASK_VALUE),
    )

    # q=1, k=0..3: causal allows k=0,1; padding allows k<3.
    # k=0: visible, k=1: visible, k=2: causal-masked, k=3: both masked.
    assert_equal(
        mask.mask(Index(0, 0, 1, 0), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    # q=2, k=4..7: all masked (both causal q<k and padding k>=3).
    assert_equal(mask.mask(Index(0, 0, 2, 4), score_vec), inf_vec)

    # q=0, k=0..3: causal allows only k=0; padding allows k<3.
    # k=0: visible, k=1..3: causal-masked.
    assert_equal(
        mask.mask(Index(0, 0, 0, 0), score_vec),
        SIMD_T(1.0, MASK_VALUE, MASK_VALUE, MASK_VALUE),
    )

    # q=2, k=0..3: causal allows k=0,1,2; padding allows k<3.
    # All three conditions agree: k=0,1,2 visible, k=3 padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 2, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, MASK_VALUE),
    )


def test_causal_padding_mask_bits() raises:
    """Test the 32-bit bitmask `mask_bits` returns when migrated to BITMASK.

    Each bit `i` (0..31) corresponds to column `col_start + i`. Bit is 1
    iff that column is visible: causal `col <= score_row` AND padding
    `col < valid_lengths[seq_id]`.
    """
    var storage: Array[UInt32, _] = [10, 3]  # valid_length per sequence
    var valid_lengths = LayoutTensor[.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    # ---- seq 0 (valid_len=10) ----
    # score_row=100, col_start=0: all 32 cols within both causal (100 >= k)
    # and padding (k < 10) => low 10 bits set, upper bits zero.
    assert_equal(
        mask.mask_bits(UInt32(0), Int32(100), Int32(0), Int32(64)),
        UInt32((1 << 10) - 1),
    )
    # score_row=100, col_start=32: all 32 cols are >= 32, so all past
    # valid_len=10 => bitmask = 0 (everything padding-masked).
    assert_equal(
        mask.mask_bits(UInt32(0), Int32(100), Int32(32), Int32(64)),
        UInt32(0),
    )
    # score_row=5, col_start=0: causal cuts at k>5 (low 6 bits set);
    # padding cuts at k>=10 (low 10 bits set). Intersection: low 6 bits.
    assert_equal(
        mask.mask_bits(UInt32(0), Int32(5), Int32(0), Int32(64)),
        UInt32((1 << 6) - 1),
    )

    # ---- seq 1 (valid_len=3) ----
    # score_row=100, col_start=0: causal allows all; padding cuts at k>=3
    # => low 3 bits set.
    assert_equal(
        mask.mask_bits(UInt32(1), Int32(100), Int32(0), Int32(64)),
        UInt32((1 << 3) - 1),
    )
    # score_row=1, col_start=0: causal cuts at k>1 (low 2 bits set);
    # padding allows up to k=2. Intersection: low 2 bits.
    assert_equal(
        mask.mask_bits(UInt32(1), Int32(1), Int32(0), Int32(64)),
        UInt32((1 << 2) - 1),
    )


def test_causal_padding_mask_multi_seq() raises:
    """Test masking with multiple sequences having different valid lengths."""
    var storage: Array[UInt32, _] = [2, 5]  # valid_length per sequence
    var valid_lengths = LayoutTensor[.uint32, Layout.row_major(2)](storage)
    var mask = CausalPaddingMask(valid_lengths)

    comptime SIMD_T = SIMD[.float32, 4]
    var score_vec = SIMD_T(1.0, 2.0, 3.0, 4.0)

    # Seq 0 (valid_length=2), q=3, k=0..3:
    # causal allows k=0,1,2,3; padding allows k<2.
    # k=0: visible, k=1: visible, k=2,3: padding-masked.
    assert_equal(
        mask.mask(Index(0, 0, 3, 0), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    # Seq 1 (valid_length=5), q=3, k=0..3:
    # causal allows k=0,1,2,3; padding allows all (k<5).
    # All visible.
    assert_equal(
        mask.mask(Index(1, 0, 3, 0), score_vec),
        SIMD_T(1.0, 2.0, 3.0, 4.0),
    )


def main() raises:
    test_causal_padding_mask_status()
    test_causal_padding_mask_set_split()
    test_causal_padding_mask_bits()
    test_causal_padding_mask_apply()
    test_causal_padding_mask_multi_seq()
