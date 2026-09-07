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

from std.collections import BitSet

from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)
from std.testing import TestSuite
from test_utils import check_write_to


def test_bitset_init() raises:
    # Test default initialization
    var bs = BitSet[128]()
    assert_equal(len(bs), 0, msg="Empty BitSet should have length 0")
    assert_false(bs, msg="Empty BitSet should be empty")

    # Test with initial bits
    var bs2 = BitSet[64]()
    assert_equal(len(bs2), 0, msg="Empty BitSet should have length 0")


def test_bitset_set_test_clear() raises:
    var bs = BitSet[128]()

    # Test setting bits
    bs.set(42)
    assert_equal(
        len(bs), 1, msg="BitSet length should be 43 after setting bit 42"
    )
    assert_true(bs.test(42), msg="Bit 42 should be set")
    assert_false(bs.test(41), msg="Bit 41 should not be set")

    # Test setting multiple bits
    bs.set(10)
    bs.set(100)
    assert_equal(
        len(bs), 3, msg="BitSet length should be 101 after setting bit 100"
    )
    assert_true(bs.test(10), msg="Bit 10 should be set")
    assert_true(bs.test(42), msg="Bit 42 should still be set")
    assert_true(bs.test(100), msg="Bit 100 should be set")

    # Test clearing bits
    bs.clear(42)
    assert_false(bs.test(42), msg="Bit 42 should be cleared")
    assert_true(bs.test(10), msg="Bit 10 should still be set")
    assert_true(bs.test(100), msg="Bit 100 should still be set")

    # Test clear_all
    bs.clear_all()
    assert_equal(len(bs), 0, msg="BitSet should be empty after clear_all")
    assert_false(bs.test(10), msg="Bit 10 should be cleared after clear_all")
    assert_false(bs.test(100), msg="Bit 100 should be cleared after clear_all")


def test_bitset_toggle() raises:
    var bs = BitSet[64]()

    # Toggle bit from 0 to 1
    bs.toggle(5)
    assert_true(bs.test(5), msg="Bit 5 should be set after toggle")

    # Toggle bit from 1 to 0
    bs.toggle(5)
    assert_false(bs.test(5), msg="Bit 5 should be cleared after second toggle")

    # Toggle multiple bits
    bs.toggle(10)
    bs.toggle(20)
    bs.toggle(30)
    assert_true(bs.test(10), msg="Bit 10 should be set")
    assert_true(bs.test(20), msg="Bit 20 should be set")
    assert_true(bs.test(30), msg="Bit 30 should be set")
    assert_equal(len(bs), 3, msg="BitSet length should be 31")


def test_bitset_toggle_all() raises:
    var bs1 = BitSet[64]()
    var bs2 = BitSet[64]()

    # set random enough pattern in both BitSets
    for idx in [0, 1, 10, 19, 22, 37, 56, 63]:
        bs1.set(idx)
        bs2.set(idx)

    # toggle all in one BitSet
    bs1.toggle_all()

    # assert that they differ in all idx
    for idx in range(64):
        assert_not_equal(
            bs1.test(idx),
            bs2.test(idx),
            msg="Bit "
            + String(idx)
            + " should be "
            + String(not bs2.test(idx))
            + " after toggle",
        )

    assert_equal(len(bs1), 56, msg="BitSet total popcount should be 56")


def test_bitset_toggle_all_non_word_length() raises:
    var bs1 = BitSet[57]()
    var bs2 = BitSet[57]()

    # set 7 bits
    for idx in [0, 1, 10, 19, 22, 37, 56]:
        bs1.set(idx)
        bs2.set(idx)

    # toggle all in one BitSet
    bs1.toggle_all()

    # assert that they differ in all idx
    for idx in range(57):
        assert_not_equal(
            bs1.test(idx),
            bs2.test(idx),
            msg="Bit "
            + String(idx)
            + " should be "
            + String(not bs2.test(idx))
            + " after toggle",
        )

    # after toggling, 50/57 bits should be set
    assert_equal(len(bs1), 50, msg="BitSet total popcount should be 50")


def test_bitset_set_all() raises:
    var bs = BitSet[64]()

    # set random enough pattern in BitSet
    for idx in [0, 1, 10, 19, 22, 37, 56, 63]:
        bs.set(idx)

    bs.set_all()

    # assert 1 in all idx
    for idx in range(64):
        assert_true(
            bs.test(idx),
            msg="Bit " + String(idx) + " should be True (1) after set all",
        )

    assert_equal(len(bs), 64, msg="BitSet total popcount should be 64")


def test_bitset_set_all_non_word_length() raises:
    var bs = BitSet[27]()

    bs.set_all()

    # assert 1 in all idx
    for idx in range(27):
        assert_true(
            bs.test(idx),
            msg="Bit " + String(idx) + " should be True (1) after set all",
        )

    assert_equal(len(bs), 27, msg="BitSet total popcount should be 27")


def test_bitset_count() raises:
    var bs = BitSet[256]()

    # Empty set should have count 0
    assert_equal(len(bs), 0, msg="Empty BitSet should have count 0")

    # Set some bits and check count
    bs.set(1)
    bs.set(10)
    bs.set(100)
    assert_equal(
        len(bs), 3, msg="BitSet should have count 3 after setting 3 bits"
    )

    # Clear a bit and check count
    bs.clear(10)
    assert_equal(
        len(bs), 2, msg="BitSet should have count 2 after clearing 1 bit"
    )

    # Clear all and check count
    bs.clear_all()
    assert_equal(len(bs), 0, msg="BitSet should have count 0 after clear_all")


def test_bitset_bounds() raises:
    var bs = BitSet[32]()

    # Test valid operations
    bs.set(0)
    bs.set(31)
    assert_true(bs.test(0), msg="Bit 0 should be set")
    assert_true(bs.test(31), msg="Bit 31 should be set")


def test_bitset_str_repr() raises:
    var bs = BitSet[16]()
    bs.set(1)
    bs.set(5)
    bs.set(10)

    var str_rep = String(bs)

    # Check that the string representations contain the set bits
    assert_true(
        "1" in str_rep, msg="String representation should contain bit 1"
    )
    assert_true(
        "5" in str_rep, msg="String representation should contain bit 5"
    )
    assert_true(
        "10" in str_rep, msg="String representation should contain bit 10"
    )
    assert_false(
        "0, " in str_rep, msg="String representation should not contain bit 0"
    )
    assert_false(
        "4, " in str_rep, msg="String representation should not contain bit 4"
    )
    assert_false(
        "16, " in str_rep, msg="String representation should not contain bit 16"
    )


def test_write_to() raises:
    var bitset = BitSet[16]()

    check_write_to(bitset, expected="{}", is_repr=False)
    check_write_to(bitset, expected="BitSet[16]({})", is_repr=True)

    bitset.set(1)
    bitset.set(5)

    check_write_to(bitset, expected="{1, 5}", is_repr=False)
    check_write_to(bitset, expected="BitSet[16]({1, 5})", is_repr=True)


def test_bitset_edge_cases() raises:
    # Test with minimum size
    var bs_min = BitSet[1]()
    bs_min.set(0)
    assert_true(
        bs_min.test(0), msg="Bit 0 should be set in minimum size BitSet"
    )
    assert_equal(
        len(bs_min),
        1,
        msg="Count should be 1 for minimum size BitSet with one bit set",
    )

    # Test with size that spans multiple words
    var bs_multi = BitSet[128]()
    bs_multi.set(63)  # Last bit in first word
    bs_multi.set(64)  # First bit in second word
    assert_true(
        bs_multi.test(63), msg="Bit 63 should be set (last bit in first word)"
    )
    assert_true(
        bs_multi.test(64), msg="Bit 64 should be set (first bit in second word)"
    )
    assert_equal(
        len(bs_multi), 2, msg="Count should be 2 for bits in different words"
    )


def test_bitset_consecutive_operations() raises:
    var bs = BitSet[64]()

    # Set and immediately clear
    bs.set(10)
    assert_true(bs.test(10), msg="Bit 10 should be set after set operation")
    bs.clear(10)
    assert_false(
        bs.test(10), msg="Bit 10 should be cleared after clear operation"
    )

    # Toggle twice to return to original state
    bs.toggle(20)
    bs.toggle(20)
    assert_false(
        bs.test(20), msg="Bit 20 should be cleared after double toggle"
    )

    # Set multiple bits and check count
    for i in range(0, 10):
        bs.set(i)
    assert_equal(len(bs), 10, msg="Count should be 10 after setting 10 bits")

    # Clear all and verify
    bs.clear_all()
    assert_equal(len(bs), 0, msg="Count should be 0 after clear_all")
    assert_false(bs, msg="BitSet should be empty after clear_all")


def test_bitset_word_boundaries() raises:
    var bs = BitSet[128]()

    # Test bits at word boundaries (assuming 64-bit words)
    bs.set(63)  # Last bit of first word
    bs.set(64)  # First bit of second word

    assert_true(
        bs.test(63), msg="Bit 63 (last bit of first word) should be set"
    )
    assert_true(
        bs.test(64), msg="Bit 64 (first bit of second word) should be set"
    )

    # Toggle bits at boundaries
    bs.toggle(63)
    bs.toggle(64)

    assert_false(bs.test(63), msg="Bit 63 should be cleared after toggle")
    assert_false(bs.test(64), msg="Bit 64 should be cleared after toggle")

    # Set bits across multiple words and check count
    for i in range(60, 70):
        bs.set(i)

    assert_equal(
        len(bs),
        10,
        msg="Count should be 10 after setting bits across word boundary",
    )


def test_bitset_large_indices() raises:
    var bs = BitSet[256]()

    # Test with larger indices
    bs.set(200)
    bs.set(250)

    assert_true(bs.test(200), msg="Bit 200 should be set")
    assert_true(bs.test(250), msg="Bit 250 should be set")
    assert_equal(
        len(bs), 2, msg="Count should be 2 after setting 2 high-index bits"
    )

    # Test string representation with large indices
    var str_rep = String(bs)
    assert_true(
        "200" in str_rep, msg="String representation should contain bit 200"
    )
    assert_true(
        "250" in str_rep, msg="String representation should contain bit 250"
    )


def test_bitset_union() raises:
    # Basic case
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.union(bs2)
    assert_equal(len(bs3), 5, msg="Union: Basic case count")
    assert_true(bs3.test(1), msg="Union: Basic case bit 1")
    assert_true(bs3.test(2), msg="Union: Basic case bit 2")
    assert_true(bs3.test(3), msg="Union: Basic case bit 3")
    assert_true(bs3.test(4), msg="Union: Basic case bit 4")
    assert_true(bs3.test(5), msg="Union: Basic case bit 5")

    # Union with empty set
    var bs_empty = BitSet[128]()
    var bs4 = bs1.union(bs_empty)
    assert_equal(len(bs4), 3, msg="Union: With empty set count")
    assert_true(bs4.test(1), msg="Union: With empty set bit 1")
    assert_true(bs4.test(2), msg="Union: With empty set bit 2")
    assert_true(bs4.test(3), msg="Union: With empty set bit 3")
    assert_false(bs4.test(4), msg="Union: With empty set bit 4")

    var bs5 = bs_empty.union(bs1)
    assert_equal(len(bs5), 3, msg="Union: Empty with non-empty set count")
    assert_true(bs5.test(1), msg="Union: Empty with non-empty set bit 1")
    assert_true(bs5.test(2), msg="Union: Empty with non-empty set bit 2")
    assert_true(bs5.test(3), msg="Union: Empty with non-empty set bit 3")

    # Union of identical sets
    var bs6 = bs1.union(bs1)
    assert_equal(len(bs6), 3, msg="Union: Identical sets count")
    assert_true(bs6.test(1), msg="Union: Identical sets bit 1")
    assert_true(bs6.test(2), msg="Union: Identical sets bit 2")
    assert_true(bs6.test(3), msg="Union: Identical sets bit 3")

    # Union of disjoint sets
    var bs7 = BitSet[128]()
    bs7.set(10)
    bs7.set(20)
    var bs8 = bs1.union(bs7)
    assert_equal(len(bs8), 5, msg="Union: Disjoint sets count")
    assert_true(bs8.test(1), msg="Union: Disjoint sets bit 1")
    assert_true(bs8.test(2), msg="Union: Disjoint sets bit 2")
    assert_true(bs8.test(3), msg="Union: Disjoint sets bit 3")
    assert_true(bs8.test(10), msg="Union: Disjoint sets bit 10")
    assert_true(bs8.test(20), msg="Union: Disjoint sets bit 20")

    # Union across word boundaries
    var bs9 = BitSet[128]()
    bs9.set(60)
    bs9.set(65)
    var bs10 = BitSet[128]()
    bs10.set(63)
    bs10.set(70)
    var bs11 = bs9.union(bs10)
    assert_equal(len(bs11), 4, msg="Union: Across words count")
    assert_true(bs11.test(60), msg="Union: Across words bit 60")
    assert_true(bs11.test(63), msg="Union: Across words bit 63")
    assert_true(bs11.test(65), msg="Union: Across words bit 65")
    assert_true(bs11.test(70), msg="Union: Across words bit 70")


def test_bitset_intersection() raises:
    # Basic case
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.intersection(bs2)
    assert_equal(len(bs3), 1, msg="Intersection: Basic case count")
    assert_true(bs3.test(3), msg="Intersection: Basic case bit 3")
    assert_false(bs3.test(1), msg="Intersection: Basic case bit 1")
    assert_false(bs3.test(2), msg="Intersection: Basic case bit 2")
    assert_false(bs3.test(4), msg="Intersection: Basic case bit 4")
    assert_false(bs3.test(5), msg="Intersection: Basic case bit 5")

    # Intersection with empty set
    var bs_empty = BitSet[128]()
    var bs4 = bs1.intersection(bs_empty)
    assert_equal(len(bs4), 0, msg="Intersection: With empty set count")

    var bs5 = bs_empty.intersection(bs1)
    assert_equal(
        len(bs5), 0, msg="Intersection: Empty with non-empty set count"
    )

    # Intersection of identical sets
    var bs6 = bs1.intersection(bs1)
    assert_equal(len(bs6), 3, msg="Intersection: Identical sets count")
    assert_true(bs6.test(1), msg="Intersection: Identical sets bit 1")
    assert_true(bs6.test(2), msg="Intersection: Identical sets bit 2")
    assert_true(bs6.test(3), msg="Intersection: Identical sets bit 3")

    # Intersection of disjoint sets
    var bs7 = BitSet[128]()
    bs7.set(10)
    bs7.set(20)
    var bs8 = bs1.intersection(bs7)
    assert_equal(len(bs8), 0, msg="Intersection: Disjoint sets count")

    # Intersection across word boundaries
    var bs9 = BitSet[128]()
    bs9.set(60)
    bs9.set(65)
    bs9.set(70)
    var bs10 = BitSet[128]()
    bs10.set(63)
    bs10.set(65)
    bs10.set(75)
    var bs11 = bs9.intersection(bs10)
    assert_equal(len(bs11), 1, msg="Intersection: Across words count")
    assert_true(bs11.test(65), msg="Intersection: Across words bit 65")
    assert_false(bs11.test(60), msg="Intersection: Across words bit 60")
    assert_false(bs11.test(63), msg="Intersection: Across words bit 63")
    assert_false(bs11.test(70), msg="Intersection: Across words bit 70")
    assert_false(bs11.test(75), msg="Intersection: Across words bit 75")


def test_bitset_difference() raises:
    # Basic case (bs1 - bs2)
    var bs1 = BitSet[128]()
    bs1.set(1)
    bs1.set(2)
    bs1.set(3)

    var bs2 = BitSet[128]()
    bs2.set(3)
    bs2.set(4)
    bs2.set(5)

    var bs3 = bs1.difference(bs2)
    assert_equal(len(bs3), 2, msg="Difference: Basic case (bs1-bs2) count")
    assert_true(bs3.test(1), msg="Difference: Basic case (bs1-bs2) bit 1")
    assert_true(bs3.test(2), msg="Difference: Basic case (bs1-bs2) bit 2")
    assert_false(bs3.test(3), msg="Difference: Basic case (bs1-bs2) bit 3")
    assert_false(bs3.test(4), msg="Difference: Basic case (bs1-bs2) bit 4")

    # Basic case (bs2 - bs1)
    var bs4 = bs2.difference(bs1)
    assert_equal(len(bs4), 2, msg="Difference: Basic case (bs2-bs1) count")
    assert_true(bs4.test(4), msg="Difference: Basic case (bs2-bs1) bit 4")
    assert_true(bs4.test(5), msg="Difference: Basic case (bs2-bs1) bit 5")
    assert_false(bs4.test(1), msg="Difference: Basic case (bs2-bs1) bit 1")
    assert_false(bs4.test(3), msg="Difference: Basic case (bs2-bs1) bit 3")

    # Difference with empty set
    var bs_empty = BitSet[128]()
    var bs5 = bs1.difference(bs_empty)
    assert_equal(len(bs5), 3, msg="Difference: With empty set count")
    assert_true(bs5.test(1), msg="Difference: With empty set bit 1")
    assert_true(bs5.test(2), msg="Difference: With empty set bit 2")
    assert_true(bs5.test(3), msg="Difference: With empty set bit 3")

    var bs6 = bs_empty.difference(bs1)
    assert_equal(len(bs6), 0, msg="Difference: Empty with non-empty set count")

    # Difference of identical sets
    var bs7 = bs1.difference(bs1)
    assert_equal(len(bs7), 0, msg="Difference: Identical sets count")

    # Difference of disjoint sets
    var bs8 = BitSet[128]()
    bs8.set(10)
    bs8.set(20)
    var bs9 = bs1.difference(bs8)  # bs1 - bs8
    assert_equal(len(bs9), 3, msg="Difference: Disjoint sets (bs1-bs8) count")
    assert_true(bs9.test(1), msg="Difference: Disjoint sets (bs1-bs8) bit 1")
    assert_true(bs9.test(2), msg="Difference: Disjoint sets (bs1-bs8) bit 2")
    assert_true(bs9.test(3), msg="Difference: Disjoint sets (bs1-bs8) bit 3")
    assert_false(bs9.test(10), msg="Difference: Disjoint sets (bs1-bs8) bit 10")

    var bs10 = bs8.difference(bs1)  # bs8 - bs1
    assert_equal(len(bs10), 2, msg="Difference: Disjoint sets (bs8-bs1) count")
    assert_true(bs10.test(10), msg="Difference: Disjoint sets (bs8-bs1) bit 10")
    assert_true(bs10.test(20), msg="Difference: Disjoint sets (bs8-bs1) bit 20")
    assert_false(bs10.test(1), msg="Difference: Disjoint sets (bs8-bs1) bit 1")

    # Difference across word boundaries
    var bs11 = BitSet[128]()
    bs11.set(60)
    bs11.set(65)
    bs11.set(70)
    var bs12 = BitSet[128]()
    bs12.set(63)
    bs12.set(65)
    bs12.set(75)
    var bs13 = bs11.difference(bs12)  # bs11 - bs12
    assert_equal(len(bs13), 2, msg="Difference: Across words (bs11-bs12) count")
    assert_true(
        bs13.test(60), msg="Difference: Across words (bs11-bs12) bit 60"
    )
    assert_true(
        bs13.test(70), msg="Difference: Across words (bs11-bs12) bit 70"
    )
    assert_false(
        bs13.test(63), msg="Difference: Across words (bs11-bs12) bit 63"
    )
    assert_false(
        bs13.test(65), msg="Difference: Across words (bs11-bs12) bit 65"
    )
    assert_false(
        bs13.test(75), msg="Difference: Across words (bs11-bs12) bit 75"
    )

    var bs14 = bs12.difference(bs11)  # bs12 - bs11
    assert_equal(len(bs14), 2, msg="Difference: Across words (bs12-bs11) count")
    assert_true(
        bs14.test(63), msg="Difference: Across words (bs12-bs11) bit 63"
    )
    assert_true(
        bs14.test(75), msg="Difference: Across words (bs12-bs11) bit 75"
    )
    assert_false(
        bs14.test(60), msg="Difference: Across words (bs12-bs11) bit 60"
    )
    assert_false(
        bs14.test(65), msg="Difference: Across words (bs12-bs11) bit 65"
    )
    assert_false(
        bs14.test(70), msg="Difference: Across words (bs12-bs11) bit 70"
    )


def test_bitset_simd_init() raises:
    var bs1 = BitSet(SIMD[.bool, 128](fill=True))
    assert_equal(len(bs1), 128, msg="BitSet count should be 128")

    var bs2 = BitSet(SIMD[.bool, 128](fill=False))
    assert_equal(len(bs2), 0, msg="BitSet count should be 0")

    var bs3 = BitSet(SIMD[.bool, 4](True, False, True, False))
    assert_equal(len(bs3), 2, msg="BitSet count should be 2")


def test_bitset_resized_from_same_size() raises:
    var src = BitSet[128]()
    src.set(1)
    src.set(64)
    src.set(127)

    var dst = BitSet[128](resized_from=src)
    assert_equal(len(dst), 3, msg="Same-size copy should preserve count")
    assert_true(dst.test(1), msg="Same-size copy: bit 1")
    assert_true(dst.test(64), msg="Same-size copy: bit 64")
    assert_true(dst.test(127), msg="Same-size copy: bit 127")

    # The copy must be independent of the source.
    dst.clear(1)
    assert_true(src.test(1), msg="Mutating the copy must not affect the source")


def test_bitset_resized_from_smaller_widens() raises:
    # Widening within a single word.
    var src = BitSet[64]()
    src.set(0)
    src.set(63)

    var dst = BitSet[128](resized_from=src)
    assert_equal(len(dst), 2, msg="Widen: count preserved")
    assert_true(dst.test(0), msg="Widen: bit 0")
    assert_true(dst.test(63), msg="Widen: bit 63")
    assert_false(dst.test(64), msg="Widen: new high bits start clear")
    assert_false(dst.test(127), msg="Widen: new high bits start clear")

    # Widening across several words must zero all the new high words.
    var narrow = BitSet[40]()
    narrow.set(0)
    narrow.set(39)

    var wide = BitSet[200](resized_from=narrow)
    assert_equal(len(wide), 2, msg="Widen across words: count preserved")
    assert_true(wide.test(0), msg="Widen across words: bit 0")
    assert_true(wide.test(39), msg="Widen across words: bit 39")
    assert_false(wide.test(100), msg="Widen across words: high word 1 clear")
    assert_false(wide.test(199), msg="Widen across words: high word 3 clear")


def test_bitset_resized_from_larger_narrows() raises:
    # Word-aligned narrowing drops whole trailing words.
    var src = BitSet[128]()
    src.set(5)
    src.set(70)  # In word 1, dropped when narrowing to a single word.

    var dst = BitSet[64](resized_from=src, truncate_set_bits=())
    assert_equal(len(dst), 1, msg="Narrow aligned: only in-range bits survive")
    assert_true(dst.test(5), msg="Narrow aligned: bit 5 survives")

    # Non-word-aligned narrowing must ignore bits at or above the new size,
    # including those sharing the last retained word.
    var big = BitSet[128]()
    big.set(10)  # In range for BitSet[40].
    big.set(39)  # In range for BitSet[40].
    big.set(40)  # Beyond 40 but in the same word.
    big.set(50)  # Beyond 40 but in the same word.
    big.set(100)  # In a dropped word.

    var small = BitSet[40](resized_from=big, truncate_set_bits=())
    assert_equal(len(small), 2, msg="Narrow unaligned: bits >= size ignored")
    assert_true(small.test(10), msg="Narrow unaligned: bit 10 survives")


def test_bitset_len() raises:
    # 1. Empty BitSet
    var bs = BitSet[128]()
    assert_equal(len(bs), 0, msg="Len: Empty BitSet should be 0")

    # 2. Single insertion
    bs.set(7)
    assert_equal(len(bs), 1, msg="Len: After setting one bit")

    # 3. Insertion across a word boundary (index 64)
    bs.set(64)
    assert_equal(len(bs), 2, msg="Len: Two bits set across word boundary")

    # 4. Toggle a set bit off
    bs.toggle(7)
    assert_equal(len(bs), 1, msg="Len: After toggling a bit off")

    # 5. Clear the remaining bit
    bs.clear(64)
    assert_equal(len(bs), 0, msg="Len: After clearing all bits")

    # 6. Bulk pattern insertion (every 3rd index)
    for i in range(128):
        if i % 3 == 0:
            bs.set(i)
    var expected = 43  # floor(128 / 3) + 1
    assert_equal(len(bs), expected, msg="Len: Pattern insertion")


def test_bitset_small_size() raises:
    # Test BitSet with size less than 64 (word size)
    var bs = BitSet[32]()
    assert_equal(len(bs), 0, msg="Small BitSet: Empty should have length 0")

    # Set a few bits
    bs.set(0)
    bs.set(15)
    bs.set(31)  # Edge of the small bitset
    assert_equal(len(bs), 3, msg="Small BitSet: Should have 3 bits set")

    # Test individual bits
    assert_true(bs.test(0), msg="Small BitSet: Bit 0 should be set")
    assert_true(bs.test(15), msg="Small BitSet: Bit 15 should be set")
    assert_true(bs.test(31), msg="Small BitSet: Bit 31 should be set")
    assert_false(bs.test(16), msg="Small BitSet: Bit 16 should not be set")

    # Test clear
    bs.clear(15)
    assert_equal(
        len(bs), 2, msg="Small BitSet: Should have 2 bits after clearing"
    )
    assert_false(bs.test(15), msg="Small BitSet: Bit 15 should be cleared")

    # Test toggle
    bs.toggle(16)
    assert_equal(
        len(bs), 3, msg="Small BitSet: Should have 3 bits after toggle"
    )
    assert_true(
        bs.test(16), msg="Small BitSet: Bit 16 should be set after toggle"
    )

    # Test clear_all
    bs.clear_all()
    assert_equal(
        len(bs), 0, msg="Small BitSet: Should be empty after clear_all"
    )
    assert_false(bs, msg="Small BitSet: Should be empty after clear_all")

    # Test very small BitSet (size 1)
    var bs1 = BitSet[1]()
    assert_equal(len(bs1), 0, msg="BitSet[1]: Empty should have length 0")
    bs1.set(0)
    assert_equal(len(bs1), 1, msg="BitSet[1]: Should have 1 bit set")
    assert_true(bs1.test(0), msg="BitSet[1]: Bit 0 should be set")

    # Test BitSet with size 2
    var bs2 = BitSet[2]()
    bs2.set(0)
    bs2.set(1)
    assert_equal(len(bs2), 2, msg="BitSet[2]: Should have 2 bits set")
    bs2.toggle(0)
    assert_equal(len(bs2), 1, msg="BitSet[2]: Should have 1 bit after toggle")
    assert_false(bs2.test(0), msg="BitSet[2]: Bit 0 should be toggled off")

    # Test BitSet with size 3 (odd size)
    var bs3 = BitSet[3]()
    bs3.set(0)
    bs3.set(1)
    bs3.set(2)
    assert_equal(len(bs3), 3, msg="BitSet[3]: Should have all 3 bits set")
    assert_true(bs3.test(2), msg="BitSet[3]: Bit 2 should be set")

    # Test BitSet with size 8 (byte boundary)
    var bs8 = BitSet[8]()
    for i in range(8):
        bs8.set(i)
    assert_equal(len(bs8), 8, msg="BitSet[8]: Should have all 8 bits set")
    bs8.clear_all()
    assert_equal(len(bs8), 0, msg="BitSet[8]: Should be empty after clear_all")

    # Test BitSet with size 63 (just under word size)
    var bs63 = BitSet[63]()
    bs63.set(62)  # Last valid bit
    assert_equal(len(bs63), 1, msg="BitSet[63]: Should have 1 bit set")
    assert_true(bs63.test(62), msg="BitSet[63]: Bit 62 should be set")

    # Test operations on small BitSets
    var bsA = BitSet[16]()
    var bsB = BitSet[16]()
    bsA.set(1)
    bsA.set(3)
    bsA.set(5)
    bsB.set(3)
    bsB.set(7)

    # Test union
    var bsUnion = bsA.union(bsB)
    assert_equal(
        len(bsUnion), 4, msg="Small BitSet union: Should have 4 bits set"
    )
    assert_true(bsUnion.test(1), msg="Small BitSet union: Bit 1 should be set")
    assert_true(bsUnion.test(3), msg="Small BitSet union: Bit 3 should be set")
    assert_true(bsUnion.test(5), msg="Small BitSet union: Bit 5 should be set")
    assert_true(bsUnion.test(7), msg="Small BitSet union: Bit 7 should be set")

    # Test intersection
    var bsIntersection = bsA.intersection(bsB)
    assert_equal(
        len(bsIntersection),
        1,
        msg="Small BitSet intersection: Should have 1 bit set",
    )
    assert_true(
        bsIntersection.test(3),
        msg="Small BitSet intersection: Bit 3 should be set",
    )

    # Test difference
    var bsDifference = bsA.difference(bsB)
    assert_equal(
        len(bsDifference),
        2,
        msg="Small BitSet difference: Should have 2 bits set",
    )
    assert_true(
        bsDifference.test(1), msg="Small BitSet difference: Bit 1 should be set"
    )
    assert_true(
        bsDifference.test(5), msg="Small BitSet difference: Bit 5 should be set"
    )
    assert_false(
        bsDifference.test(3),
        msg="Small BitSet difference: Bit 3 should not be set",
    )


def test_bitset_test_all_before_after_single_word() raises:
    # Bits 0..9 set, rest clear (within a single word).
    var bs = BitSet[64]()
    for i in range(10):
        bs.set(i)

    # Everything below the first clear bit is set.
    assert_true(bs.test_range[True, hi=10](), msg="[0,10) all set")
    assert_true(bs.test_range[True, hi=5](), msg="[0,5) all set")
    assert_true(bs.test_range[True, hi=0](), msg="empty range is vacuously set")
    assert_false(bs.test_range[True, hi=11](), msg="bit 10 is clear")

    # Everything at or below the last set bit is not all clear.
    assert_false(bs.test_range[False, lo=9](), msg="(9,64) all clear")
    assert_true(bs.test_range[False, lo=10](), msg="(10,64) all clear")
    assert_false(bs.test_range[False, lo=8](), msg="bit 9 is set")
    assert_true(
        bs.test_range[False, lo=63](), msg="empty range is vacuously clear"
    )

    # Inverse predicates.
    assert_false(bs.test_range[False, hi=5](), msg="[0,5) not clear")
    assert_true(bs.test_range[False, hi=0](), msg="empty range vacuously clear")
    assert_false(bs.test_range[True, lo=9](), msg="(9,64) not all set")


def test_bitset_test_all_across_words() raises:
    # Set every bit in [0, 200); leave [200, 256) clear across word boundaries.
    var bs = BitSet[256]()
    for i in range(200):
        bs.set(i)

    assert_true(bs.test_range[True, hi=200](), msg="[0,200) all set")
    assert_true(bs.test_range[True, hi=128](), msg="[0,128) all set")
    assert_true(bs.test_range[True, hi=65](), msg="[0,65) all set")
    assert_false(bs.test_range[True, hi=201](), msg="bit 200 is clear")

    assert_false(bs.test_range[False, lo=199](), msg="(199,256) all clear")
    assert_true(bs.test_range[False, lo=200](), msg="(200,256) all clear")
    assert_false(bs.test_range[False, lo=198](), msg="bit 199 is set")

    # A single clear bit in the middle breaks a run of set bits.
    bs.clear(100)
    assert_false(bs.test_range[True, hi=200](), msg="bit 100 now clear")
    assert_true(bs.test_range[True, hi=100](), msg="[0,100) still all set")


def test_bitset_test_all_full_and_empty() raises:
    var full = BitSet[130]()
    full.set_all()
    assert_true(
        full.test_range[True, hi=130](), msg="fully-set: all before set"
    )
    assert_true(full.test_range[True, lo=0](), msg="fully-set: all after set")

    var empty = BitSet[130]()
    assert_true(
        empty.test_range[False, hi=130](), msg="empty: all before clear"
    )
    assert_true(empty.test_range[False, lo=0](), msg="empty: all after clear")


def test_bitset_test_all_large_set() raises:
    # 2048 bits (32 words) so the interior-word SIMD loop runs several full
    # batches plus a scalar drain regardless of `simd_width_of[Int64]()`
    # (2 on NEON, 8 on AVX-512). The `[0, 650)` span has ~9 interior words.
    comptime N = 2048
    var bs = BitSet[N]()
    bs.set_all()

    assert_true(bs.test_range[True, hi=650](), msg="large: [0,650) all set")
    assert_true(bs.test_range[True, lo=3](), msg="large: (3,N) all set")
    assert_true(bs.test_range[True, lo=63](), msg="large: (63,N) all set")
    assert_false(bs.test_range[False, hi=650](), msg="large: [0,650) not clear")
    assert_false(bs.test_range[False, lo=3](), msg="large: (3,N) not clear")

    # A single clear bit deep in an interior word must break an all-set run
    # scanned from either direction.
    bs.clear(300)  # Word 4 -- an interior word for both spans below.
    assert_false(bs.test_range[True, hi=650](), msg="large: before-set hole")
    assert_true(bs.test_range[True, hi=300](), msg="large: [0,300) still set")
    assert_false(bs.test_range[True, lo=3](), msg="large: after-set hole")
    assert_true(bs.test_range[True, lo=301](), msg="large: (300,N) still set")


def test_bitset_test_all_large_unset() raises:
    comptime N = 2048
    var bs = BitSet[N]()  # All clear.

    assert_true(bs.test_range[False, hi=650](), msg="large: [0,650) all clear")
    assert_true(bs.test_range[False, lo=3](), msg="large: (3,N) all clear")
    assert_false(bs.test_range[True, hi=650](), msg="large: [0,650) not set")

    # A single set bit deep in an interior word breaks an all-clear run.
    bs.set(300)
    assert_false(bs.test_range[False, hi=650](), msg="large: before-unset hole")
    assert_true(
        bs.test_range[False, hi=300](), msg="large: [0,300) still clear"
    )
    assert_false(bs.test_range[False, lo=3](), msg="large: after-unset hole")
    assert_true(
        bs.test_range[False, lo=301](), msg="large: [301,N) still clear"
    )


def test_bitset_test_all_min_size() raises:
    # In a size-1 set the only valid index is 0, so every range around it is
    # empty and therefore vacuously true for all predicates.
    var one = BitSet[1]()
    one.set(0)
    assert_true(one.test_range[True, hi=0](), msg="size-1: empty before set")
    assert_true(one.test_range[True, lo=0](), msg="size-1: empty after set")
    assert_true(one.test_range[False, hi=0](), msg="size-1: empty before unset")
    assert_true(one.test_range[False, lo=1](), msg="size-1: empty after unset")


def test_bitset_resized_from_large_simd() raises:
    # 31 words (odd) so the copy loop runs full SIMD batches plus a scalar
    # drain on any width; the widen target adds an odd number of zero words so
    # the zero loop also hits a drain.
    comptime SRC = 1984  # 31 words.
    var src = BitSet[SRC]()
    for i in range(SRC):
        if i % 3 == 0:
            src.set(i)
    comptime expected = SRC // 3 + 1

    # Same-size large copy: every word must match bit-for-bit.
    var same = BitSet[SRC](resized_from=src, truncate_set_bits=())
    assert_equal(len(same), expected, msg="large copy: count matches")
    for i in range(SRC):
        assert_equal(same.test(i), src.test(i), msg="large copy: bit matches")

    # Widen to 48 words (17 new zero words).
    var wide = BitSet[3072](resized_from=src, truncate_set_bits=())
    assert_equal(len(wide), expected, msg="large widen: count preserved")
    for i in range(SRC):
        assert_equal(
            wide.test(i), src.test(i), msg="large widen: low bits match"
        )
    assert_false(wide.test(SRC), msg="large widen: first new word cleared")
    assert_false(wide.test(3071), msg="large widen: last new word cleared")

    # Narrow to 16 words: bits at or above the new size are dropped, so the
    # count reflects only set bits in `[0, 1000)`.
    var small = BitSet[1000](resized_from=src, truncate_set_bits=())
    for i in range(1000):
        assert_equal(
            small.test(i), src.test(i), msg="large narrow: bit matches"
        )
    comptime narrow_expected = 999 // 3 + 1
    assert_equal(
        len(small), narrow_expected, msg="large narrow: no bits leak past size"
    )


def test_bitset_resized_from_edge_contents() raises:
    # Empty source widened stays empty.
    var empty = BitSet[64]()
    var widened_empty = BitSet[256](resized_from=empty, truncate_set_bits=())
    assert_equal(len(widened_empty), 0, msg="empty widened stays empty")

    # Full source narrowed fills exactly the retained range; the bits beyond
    # the new size in the last retained word must be masked off.
    var full = BitSet[256]()
    full.set_all()
    var narrowed_full = BitSet[100](resized_from=full, truncate_set_bits=())
    assert_equal(len(narrowed_full), 100, msg="full narrowed fills new range")

    # Independence must hold after widening and narrowing, not just same-size.
    var src = BitSet[128]()
    src.set(5)

    var w = BitSet[256](resized_from=src, truncate_set_bits=())
    w.set(200)
    assert_equal(
        len(src), 1, msg="source unchanged after mutating widened copy"
    )

    var n = BitSet[64](resized_from=src, truncate_set_bits=())
    n.set(10)
    assert_equal(
        len(src), 1, msg="source unchanged after mutating narrowed copy"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
